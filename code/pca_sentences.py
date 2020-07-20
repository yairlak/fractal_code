#!/usr/bin/env python
import os, pickle, json
import argparse
import itertools
import numpy as np
from tqdm import tqdm
from sklearn import decomposition
from sklearn import preprocessing
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Evaluate pt model on an agreement task')
parser.add_argument('--model', type=str, default='../data/models/model_LSTM_p1_0.5_p2_0.5_nwords_1000000_length_clip_21_depth_clip_5_seed1_LSTM_dropout_0.1_nlayers_2_nhid_32_bptt_length_32_embsz_8.pt', help='path to pt model - after training')
parser.add_argument('--sentences', default=[], type=str, help='Path to text file containing the list of sentences to analyze')
parser.add_argument('--meta', default=[], type=str, help='The corresponding meta data of the sentences')
parser.add_argument('--activations', default=[] ,type=str, help='The corresponding sentence (LSTM) activations')
parser.add_argument('--activation-type', default='hidden',type=str, help='hidden/cell')
parser.add_argument('--p1', type=float, default=0.5)
parser.add_argument('--p2', type=float, default=0.5)
parser.add_argument('--num_units', default=[], type=int, help='if model is not given then this will be determined from the best_model from the grid search')
parser.add_argument('--depth', default=3, type=int)
parser.add_argument('--length', default=2, type=int)
parser.add_argument('--num-components', default=4, type=int)
# parser.add_argument('--vocabulary', default='', type=str, help='path to vocabulary. If empty then vocab extension is added to the filename of the model')
parser.add_argument('--noun-initial', default='n', help='The character with which noun tokens begin. Results will include agreement performance on all verbs')
parser.add_argument('--verb-initial', default='v', help='The character with which verb tokens begin. Results will include agreement performance on all verbs')
parser.add_argument('--feature-values', default=['sg', 'pl'], help='(list of of str) The corresponding feature values in the input and results vectors')
parser.add_argument('--markzone', action='store_true', default=False)
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--save_fn', default=[], type=str ,help='path to results folder')
args = parser.parse_args()


if not args.sentences:
    args.sentences = '../data/datasets/test_depth_%i_length_%i.txt' % (args.depth, args.length)
    args.meta = '../data/datasets/test_depth_%i_length_%i.info' % (args.depth, args.length)

if not args.activations:
    args.activations = '../results/activations/%s_%s.pkl' % (os.path.basename(args.model), os.path.basename(args.sentences))


fn = 'pca_1d_%s_depth_%i_length_%i_%s' % (args.activation_type, args.depth, args.length, os.path.basename(args.model))
if not args.save_fn:
    args.save_fn = os.path.join('..', 'figures', 'pca', fn)
os.makedirs(os.path.dirname(args.save_fn), exist_ok=True)

print('Path to model: ' + args.model)
print('Path to input sentences: ' + args.sentences)
print('Path to model activations: ' + args.activations)

def get_verb_indexes_in_onehot_encoding(vocab, verb_initial, features):
    '''
    !!! Assumes that the features are ['sg', 'pl'] !!!
    '''
    assert features[0] == 'sg' and features[1] == 'pl'
    word2idx = vocab.word2idx

    vocab_token_strings = word2idx.keys()
    singular_verb_token_strings = [w for w in vocab_token_strings if w[0] == verb_initial and (features[0] in w)]
    plural_verb_token_strings = [w for w in vocab_token_strings if w[0] == verb_initial and (features[1] in w)]

    return [word2idx[v] for v in singular_verb_token_strings], [word2idx[v] for v in plural_verb_token_strings]



def get_agreements_center_embedding(sentence, noun_initial, verb_initial):
    '''
    sentence - (list of strings) The sentence to analyze as a list of its word strings.
    noun_initial - (str) how to identify a noun token by its FIRST letter (e.g., 'n' for 'n1', 'n2',...)
    verb_initial - (str) how to identify a verb token by its FIRST letter (e.g., 'v' for 'v1', 'v2',...)

    Returns:
    agreements - (dict) each key is an index of a noun word in the sentence (IX_noun) and its value is a
    tuple with the corresponding verb index and features (IX_verb, features_of_noun). The features are a
    list of features (e.g., ['sg', 'm']).
    IX_nouns - (list) indexes to noun words in the sentences (counting from ZERO).
    IX_verbs - (list) indexes to verb words in the sentences (counting from ZERO).
    '''
    agreements = {}
    IX_nouns = [i for (i, w) in enumerate(sentence) if w[0] == noun_initial]
    IX_verbs = [i for (i, w) in enumerate(sentence) if w[0] == verb_initial]
    assert len(IX_nouns) == len(IX_verbs) # check that there's an equal number of nouns and verbs in the sentence

    for IX_noun, IX_verb in zip(IX_nouns, reversed(IX_verbs)):
        features_of_noun = sentence[IX_noun][1+sentence[IX_noun].find('['):sentence[IX_noun].find(']')].split(',')
        features_of_noun = [f.strip() for f in features_of_noun]
        features_of_verb = sentence[IX_verb][1+sentence[IX_verb].find('['):sentence[IX_verb].find(']')].split(',')
        features_of_verb = [f.strip() for f in features_of_verb]
        assert features_of_noun == features_of_verb
        agreements[IX_noun] = (IX_verb, features_of_noun)
        agreements[IX_verb] = (IX_noun, features_of_verb)


    return agreements, IX_nouns, IX_verbs


def get_wrong_prediction(correct_word):

    correct_feature = correct_word[1+correct_word.find('['):correct_word.find(']')].strip()
    pos = correct_word[:correct_word.find('[')]
    wrong_feature = set(args.feature_values)- set([correct_feature])
    assert len(wrong_feature)==1
    wrong_word = pos + '[' + list(wrong_feature)[0] + ']'

    return wrong_word


############
# MAIN #####
############

# ----------------------
# --- Load text data ---
# ----------------------
sentences_str = [l.rstrip('\n').split(' ') for l in open(args.sentences, encoding='utf-8')]


# ----------------------------------------
# --- Load unit and output activations ---
# ----------------------------------------
print('Loading activations...')
LSTM_activations = pickle.load(open(args.activations, 'rb'))
LSTM_activations = LSTM_activations[args.activation_type]

# Reshape: concatenate all timepoints before PCAing (num_trials*num_timepoints X num_features)
print('Prepare data for PCA: breaking down sentences into words')
design_matrix_all_words = np.vstack([np.reshape(word_data, (1, -1), order='C') for sentence_data in tqdm(LSTM_activations) for word_data in sentence_data])

# print('Prepare data for PCA: standardize data')
# standardized_scale = preprocessing.StandardScaler().fit(design_matrix_all_words)
# words_standardized = standardized_scale.transform(design_matrix_all_words)


print('Run PCA')
pca = decomposition.PCA(n_components=args.num_components)
pca.fit(design_matrix_all_words)
words_PCA_projected = pca.transform(design_matrix_all_words)

np.savetxt('../results/activations/pca/' + fn + '.csv', pca.components_, delimiter=",")


PCA_trajectories = []
st = 0
for _, sentence_data in enumerate(tqdm(LSTM_activations)):
    curr_sentence_data = []
    for _ in sentence_data:
        curr_sentence_data.append(words_PCA_projected[st, :])
        st += 1
    PCA_trajectories.append(np.asarray(curr_sentence_data).transpose()) # list (len=num_trials) of ndarrays (num_timepoints X num_PC_components)

def generate_figure_for_PC_trajectory(PCs, explained_variance_ratio_, curr_stimuli,
                                      vectors_pca_trajectories_mean_over_structure,
                                      vectors_pca_trajectories_std_structure, labels):

    plt.scatter(vectors_pca_trajectories_mean_over_structure[PCs[0], :],
                  vectors_pca_trajectories_mean_over_structure[PCs[1], :], label=labels)
    plt.errorbar(vectors_pca_trajectories_mean_over_structure[PCs[0], :],
                   vectors_pca_trajectories_mean_over_structure[PCs[1], :],
                   xerr=vectors_pca_trajectories_std_structure[PCs[0], :],
                   yerr=vectors_pca_trajectories_std_structure[PCs[1], :], linewidth=5)

    # Annotate with number the subsequent time points on the trajectories
    delta_x = 0.03  # Shift the annotation of the time point by a small step
    sent_str = ''
    for timepoint in range(vectors_pca_trajectories_mean_over_structure.shape[1]):
        curr_token = curr_stimuli[0][timepoint][0] + curr_stimuli[0][timepoint][2:]
        sent_str += curr_token[0] + ' '
        if 'n' in curr_token or 'v' in curr_token or timepoint == 0:
            plt.annotate(str(timepoint + 1) + ':' + curr_token, xy=(
            delta_x + vectors_pca_trajectories_mean_over_structure[PCs[0], timepoint],
            delta_x + vectors_pca_trajectories_mean_over_structure[PCs[1], timepoint]), fontsize=25)

    # plt.legend()
    plt.xlabel('PC %i %1.2f' % (PCs[0] + 1, explained_variance_ratio_[PCs[0]]), fontsize=26)
    plt.ylabel('PC %i %1.2f' % (PCs[1] + 1, explained_variance_ratio_[PCs[1]]), fontsize=26)
    plt.tick_params(labelsize=30)
    plt.title(sent_str, fontsize=20)
    # plt.xlim([-12, 12])
    # plt.ylim([-12, 12])



def get_IX_to_structures():
    IX_structures = []
    structures_names = []

    with open(args.meta, "rb") as info_file:
        meta = json.load(info_file)

    lst = list(itertools.product([0, 1], repeat=len(meta[0])))

    for c, condition in enumerate(lst):
        IX_structure = []
        structure_name = ''
        for i, m in enumerate(meta):
            fullfill_condition = True
            for v, sg_pl_binary in enumerate(condition):
                sg_pl_string = 'singular' if sg_pl_binary == 0 else 'plural'
                if i == 0:
                    structure_name += sg_pl_string[0].upper()
                if not m['number_%i' % (v + 1)] == sg_pl_string:
                    fullfill_condition = False
            if fullfill_condition:
                IX_structure.append(i)
        structures_names.append(structure_name)
        IX_structures.append(IX_structure)

    return IX_structures, structures_names

IX_structures, structures_names = get_IX_to_structures()
num_structures = len(IX_structures)

print('After PCA: average across each syntactic structure type')
vectors_pca_trajectories_mean_over_structure = []
vectors_pca_trajectories_std_structure = []
for i, IX_structure in enumerate(IX_structures):
    vectors_of_curr_structure = [vec for ind, vec in enumerate(PCA_trajectories) if ind in IX_structure]
    # print(i, len(vectors_of_curr_structure ))
    vectors_pca_trajectories_mean_over_structure.append(np.mean(np.asarray(vectors_of_curr_structure), axis=0))
    vectors_pca_trajectories_std_structure.append(np.std(np.asarray(vectors_of_curr_structure), axis=0))

# Plot averaged trajectories for all structures
# for PCs in itertools.combinations(range(args.num_components), 2):
###############
# 1d plots
##############
# a = int(np.ceil(num_structures ** (0.5)))
# b = int(np.ceil(num_structures/a))
ls = '--'
lw = '3'
color = 'b'
n_rows = int(np.floor(np.sqrt(args.num_components)))
n_cols = int(np.ceil(args.num_components / n_rows))
fig, axarr = plt.subplots(n_rows, n_cols, figsize=(n_cols * 10, n_rows * 6))
for curr_pc, ax in enumerate(axarr.flat):
    print('PCA %i' % curr_pc)
    for i in tqdm(range(num_structures)):
        # plt.axes(axarr.flat[i])
        if len(structures_names[i]) > 2:
            if structures_names[i][0] == 'S':
                color = 'r'
            else:
                color = 'b'
            if structures_names[i][1] == 'S':
                ls = '--'
            else:
                ls = '-'
            if structures_names[i][2] == 'S':
                lw = 6
            else:
                lw = 3

            if IX_structures[i] and curr_pc < args.num_components:
                print(i, structures_names[i])
                curr_stimuli = [stim for ind, stim in enumerate(sentences_str) if ind in IX_structures[i]]
                x = range(len(vectors_pca_trajectories_mean_over_structure[i][curr_pc]))
                y = vectors_pca_trajectories_mean_over_structure[i][curr_pc]
                yerr = vectors_pca_trajectories_std_structure[i][curr_pc]
                ax.errorbar(x, y, yerr=yerr, lw=lw, ls=ls, color=color, label=structures_names[i])


                # sent_str = ''
                # for timepoint in range(vectors_pca_trajectories_mean_over_structure[0].shape[1]):
                #     curr_token = curr_stimuli[0][timepoint][0] + curr_stimuli[0][timepoint][2:]
                #     sent_str += curr_token[0] + ' '
                # sent_str = sent_str.split(' ')
                #
                # # ax.legend()
                # ax.set_xticks(range(len(sent_str)))
                # ax.set_xticklabels(sent_str)
                words = sentences_str[0]
                words = [w[0] for w in words]
                ax.set_xticks(range(len(words)))
                ax.set_xticklabels(words)
                ax.set_xlabel('Time', fontsize=26)
                ax.set_ylabel('PC value', fontsize=26)
                ax.tick_params(labelsize=20)
                ax.set_title('PC %i' % (curr_pc+1), fontsize=20)
                if args.markzone:
                    ax.axvspan(9, 10, facecolor='yellowgreen', alpha=0.9)
                    ax.axvspan(12, 13, facecolor='yellowgreen', alpha=0.9)
                    ax.axvspan(15, 16, facecolor='yellowgreen', alpha=0.9)
                # plt.xlim([-12, 12])
                # plt.ylim([-12, 12])
plt.tight_layout()
plt.figure(fig.number)
plt.savefig(args.save_fn + '.png')
plt.close(fig.number)
print('Figure saved to: %s' % args.save_fn)
