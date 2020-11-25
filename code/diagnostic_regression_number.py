#!/usr/bin/env python
import os, pickle, json
import argparse
from viz import plot_GAT, plot_weight_trajectory
import numpy as np
import matplotlib.pyplot as plt
import sklearn, mne
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


parser = argparse.ArgumentParser(description='Ablate and eval model on test dataset')
parser.add_argument('--model', type=str, default='../data/models/model_LSTM_p1_0.5_p2_0.5_nwords_1000000_length_clip_21_depth_clip_5_seed1_LSTM_dropout_0.1_nlayers_2_nhid_32_bptt_length_32_embsz_8.pt', help='path to trained pt model. If empty list then best model is taken based on p1 and p2.')
parser.add_argument('--sentences', type=str, default=[], help='Path to text file containing the list of sentences to analyze')
parser.add_argument('--meta', type=str, default=[], help='The corresponding meta data of the sentences')
parser.add_argument('--var-type', type=str, default='hidden', choices=['hidden', 'cell'])
parser.add_argument('--p1', type=float, default=0.5)
parser.add_argument('--p2', type=float, default=0.5)
parser.add_argument('--depth', default=3, type=int)
parser.add_argument('--length', default=2, type=int)
parser.add_argument('--unit', default=0, type=int)
parser.add_argument('--unit-to-plot', default=23, type=int)
parser.add_argument('--relative-pert', action='store_true', default=True)
parser.add_argument('--pc', type=int, default=1, help='If args.unit=0 (pertubation in pc direction), then this arg determines which PC to take (counting from 1!).')
parser.add_argument('--currents', type=float, default=[-10, -5, 0, 1, 10])
parser.add_argument('--t-for-activation-plot', default=[2, 4, 6], type=int)
args = parser.parse_args()
print(sklearn.__version__, mne.__version__)


########
# Load #
########

if not args.sentences:
    args.sentences = '../data/datasets/test_depth_%i_length_%i.txt' % (args.depth, args.length)
    args.meta = '../data/datasets/test_depth_%i_length_%i.info' % (args.depth, args.length)

# Load meta data and generate indexing for each verb, whether it is singular or plural
with open(args.meta, "rb") as info_file:
    info = json.load(info_file)

with open(args.sentences, 'r') as f_sentences:
    sentences = f_sentences.readlines()

IX2sentences = {}
for v in range(args.depth):
    IX2sentences[v+1] = {}
    IX2sentences[v + 1]['singular'] = [i for i, s in enumerate(info) if s['number_%i' % (v + 1)] == 'singular']
    IX2sentences[v + 1]['plural'] = [i for i, s in enumerate(info) if s['number_%i' % (v + 1)] == 'plural']

print('Loading activations...')
activations = '../results/activations/%s_%s.pkl' % (os.path.basename(args.model), os.path.basename(args.sentences))
LSTM_activations = pickle.load(open(activations, 'rb'))
LSTM_activations = LSTM_activations[args.var_type]

num_trials, num_units, num_timepoints = len(LSTM_activations), LSTM_activations[0][0].size, len(LSTM_activations[0])
num_words = len(sentences[0].split(' '))
assert num_words==num_timepoints
words = sentences[0].split(' ')
words = [w[0] for w in words]


#######
# GAT #
#######

# PREPARE DATA (X, y)
X = []
for i_trial, trial_data in enumerate(LSTM_activations):
    trial_vector_t = []
    for i_t, t_data in enumerate(trial_data):
        trial_vector_t.append(t_data.reshape(-1)) # flatten multiple layers into a single one
    X.append(np.vstack(trial_vector_t).transpose())
X = np.asarray(X)

# dict with keys for grammatical number (1,2,..), which in turn contain list (len=#cv-splits) of sublists (len=#timepoints).
mean_activation = {}
for v in range(3):
    mean_activation[v+1] = {}
weights_clf = {}
diag_scores = {}
for v in range(args.depth):
    weights_clf[v + 1] = {}
    weights_clf[v + 1]['splits'] = []
    IX_sing = IX2sentences[v + 1]['singular']
    IX_plur = IX2sentences[v + 1]['plural']
    mean_activation[v + 1]['singular'] = np.mean(X[IX_sing, :, :], axis=0)
    mean_activation[v + 1]['plural'] = np.mean(X[IX_plur, :, :], axis=0)
    mean_activation[v + 1]['difference'] = np.mean(X[IX_sing, :, :], axis=0) - np.mean(X[IX_plur, :, :], axis=0)
    Y = np.zeros(num_trials)
    Y[IX_sing] = 1
    Y[IX_plur] = 2
    del IX_sing, IX_plur
    assert all(Y>0)
    print(list(Y).count(1), list(Y).count(2))

    # clf = make_pipeline(StandardScaler(), LinearSVC(class_weight='balanced'))
    clf = make_pipeline(LinearSVC(class_weight='balanced'))
    time_gen = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=-1, verbose=True)
    cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    #cv = StratifiedShuffleSplit(n_splits=20, random_state=0)


    scores = []
    for i, (train, test) in enumerate(cv.split(X, Y)):
        time_gen.fit(X[train], Y[train])
        scores.append(time_gen.score(X[test], Y[test]))
        # list (len=#cv-splits) of sublists (len=#timepoints):
        curr_weights_clf = np.asarray([np.squeeze(w._final_estimator.coef_) for w in time_gen.estimators_])
        weights_clf[v + 1]['splits'].append(curr_weights_clf)

    weights_clf[v + 1]['mean'] = np.mean(np.asarray(weights_clf[v + 1]['splits']), axis=0)
    weights_clf[v + 1]['std'] = np.std(np.asarray(weights_clf[v + 1]['splits']), axis=0)

    ########
    # PLOT #
    ########
    scores = np.mean(scores, axis=0)
    diag_scores[v+1] = np.diagonal(scores)
    fig, axs = plot_GAT(scores, v + 1, words, f'Number {v+1}; {args.var_type}')
    fname = f'GAT_{args.var_type}_number_{v+1}.png'
    plt.savefig(f'../figures/gat/{fname}')
    print(f'Saved to: ../figures/gat/{fname}')




########################
# PCA MEAN ACTIVATIONS #
########################
for dim in [2, 3]: # 2d and 3d PCA plots
    pca = PCA(n_components=dim)
    X = []; num_timepoints = []
    for v in range(3):
        X.append(mean_activation[v + 1]['difference'].transpose())
        num_timepoints.append(mean_activation[v + 1]['difference'].transpose().shape[0])
    X = np.vstack(X) # lump vectos across grammatical number
    mean_activation_projected = pca.fit_transform(X)
    print(f'Explained Variance ({dim}d): {pca.explained_variance_ratio_}')
    # mean_activation_projected = pca.transform(mean_activation[v + 1]['difference'].transpose())

    # Separate according to grammatical number
    mean_activation_projected_dict = {}
    for v in range(args.depth):
        st = v*num_words
        ed = (v+1)*num_words
        mean_activation_projected_dict[v+1] = mean_activation_projected[st:ed,:]
    # PLOT
    for v_highlight in range(args.depth):
        fig_pca, ax_pca = plot_weight_trajectory(mean_activation_projected_dict, mean_activation, diag_scores, words, f'{dim}d', v_highlight)
        if dim == 2:
            x_origin, y_origin = pca.transform(np.zeros((1, X.shape[1])))[0]
            # x_origin, y_origin = 0, 0
            ax_pca.axhline(y=y_origin, color='k')
            ax_pca.axvline(x=x_origin, color='k')
            ax_pca.scatter(x_origin, y_origin)

        elif dim==3:
            x_origin, y_origin, z_origin = pca.transform(np.zeros((1, X.shape[1])))[0]

        # ax_pca.set_xlim([-1.2, 1])
        # ax_pca.set_ylim([-0.6, 1])
        # SAVE
        fname = f'PCA{dim}D_mean_activations_number_{v_highlight+1}_{args.var_type}.png'
        fig_pca.savefig(f'../figures/gat/{fname}')
        print(f'Saved to: ../figures/gat/{fname}')


weights_all_numbers = np.vstack([weights_clf[v]['mean'] for v in range(1,4)])
# weights_std_all_numbers = np.vstack([weights_clf[v + 1]['std'] for v in range(1,4)])
#############################
# PCA WEIGHTS OF CLASSIFIER #
#############################
for dim in [2, 3]: # 2d and 3d PCA plots
    pca = PCA(n_components=dim)
    pca.fit(weights_all_numbers)
    print(f'Explained Variance ({dim}d): {pca.explained_variance_ratio_}')
    weights_projected = pca.transform(weights_all_numbers)
    mean_activation_projected = {}
    for v in range(3):
        mean_activation_projected[v+1] = {}
        for sp in ['singular', 'plural']:
            mean_activation_projected[v+1][sp] = pca.transform(mean_activation[v+1][sp].transpose())

    # Separate according to grammatical number
    weights_projected_dict = {}
    for v in range(args.depth):
        st = v*num_words
        ed = (v+1)*num_words
        weights_projected_dict[v+1] = weights_projected[st:ed,:]
    # PLOT
    for v_highlight in range(args.depth):
        fig_pca, ax_pca = plot_weight_trajectory(weights_projected_dict, mean_activation_projected, diag_scores, words, f'{dim}d', v_highlight)
        if dim == 2:
            x_origin, y_origin = pca.transform(np.zeros((1, weights_all_numbers.shape[1])))[0]
            x_origin, y_origin = 0, 0
            ax_pca.axhline(y=y_origin, color='k')
            ax_pca.axvline(x=x_origin, color='k')
            ax_pca.scatter(x_origin, y_origin)
        elif dim==3:
            x_origin, y_origin, z_origin = pca.transform(np.zeros((1, weights_all_numbers.shape[1])))[0]

        ax_pca.set_xlim([-1.2, 1])
        ax_pca.set_ylim([-0.6, 1])
        # SAVE
        fname = f'PCA{dim}D_weights_number_{v_highlight+1}_{args.var_type}.png'
        fig_pca.savefig(f'../figures/gat/{fname}')
        print(f'Saved to: ../figures/gat/{fname}')


# PLOT ALL WEIGHT on PC1 and PC2
number_colors = {1: 'r', 2: 'g', 3: 'b'}
ls = {1: '-', 2: '--'}
fig_both_pc, ax_both_pc = plt.subplots(figsize=(10, 10))
for pc in range(1,3):
    fig_pc, ax_pc = plt.subplots(figsize=(10, 10))
    for v in range(3):
        # PLOT PC of weights
        ax_pc.plot(range(1, len(words) + 1), weights_projected_dict[v + 1][:, pc-1], color=number_colors[v + 1], label='V'+str(v+1))
        ax_both_pc.plot(range(1, len(words) + 1), weights_projected_dict[v + 1][:, pc - 1], color=number_colors[v + 1],
                        label=f'V{v+1}-PC{pc}', ls=ls[pc])

    ax_pc.set_xlabel('Times')
    ax_pc.set_ylabel(f'PC{pc}')
    ax_pc.set_xticks(range(1, len(words) + 1))
    ax_pc.set_xticklabels(words)
    ax_pc.legend(loc=1)
    ax_pc.set_ylim([-2, 2])

    fname = f'PC{pc}_weights_{args.var_type}.png'
    fig_pc.savefig(f'../figures/gat/{fname}')
    print(f'Saved to: ../figures/gat/{fname}')

for x in range(3,21,3):
    ax_both_pc.axvline(x, ls='--', color='k', lw=1)
ax_both_pc.set_xlabel('Times')
ax_both_pc.set_ylabel(f'PC{pc}')
ax_both_pc.set_xticks(range(1, len(words) + 1))
ax_both_pc.set_xticklabels(words)
ax_both_pc.legend(loc=1)
ax_both_pc.set_ylim([-2, 2])
fname = f'PC_both_weights_{args.var_type}.png'
fig_both_pc.savefig(f'../figures/gat/{fname}')
print(f'Saved to: ../figures/gat/{fname}')