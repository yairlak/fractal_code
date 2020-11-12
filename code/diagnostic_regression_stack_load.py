#!/usr/bin/env python
import os, argparse, pickle, json
import argparse
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV, RidgeCV

parser = argparse.ArgumentParser(description='Ablate and eval model on test dataset')
parser.add_argument('--model', type=str, default='../data/models/model_LSTM_p1_0.5_p2_0.5_nwords_1000000_length_clip_21_depth_clip_5_seed1_LSTM_dropout_0.1_nlayers_2_nhid_32_bptt_length_32_embsz_8.pt', help='path to trained pt model. If empty list then best model is taken based on p1 and p2.')
parser.add_argument('--sentences', type=str, default=[], help='Path to text file containing the list of sentences to analyze')
parser.add_argument('--meta', type=str, default=[], help='The corresponding meta data of the sentences')
parser.add_argument('--var-type', type=str, default='hidden', choices=['hidden', 'cell'])
parser.add_argument('--depth', default=3, type=int)
parser.add_argument('--length', default=2, type=int)
args = parser.parse_args()


##############
# Load stuff #
##############

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
activations_sentences = LSTM_activations['sentences']
LSTM_activations = LSTM_activations[args.var_type]

num_trials, num_units, num_timepoints = len(LSTM_activations), LSTM_activations[0][0].size, len(LSTM_activations[0])
num_words = len(sentences[0].split(' '))
assert num_words==num_timepoints
words = sentences[0].split(' ')
words = [w[0] for w in words]

###################
# CALC STACK LOAD #
###################
dict_stack_load = {}
cnt_stack_load = 0
for i_token, token in enumerate(activations_sentences[0,:]):
    dict_stack_load[i_token] = cnt_stack_load
    if token[0] == 'n':
        cnt_stack_load+=1
    elif token[0] == 'v':
        cnt_stack_load-=1
X = []; y = []
for i_trial, trial_data in enumerate(LSTM_activations):
    trial_vector_t = []
    for i_t, t_data in enumerate(trial_data):
        X.append(t_data.flatten())
        y.append(dict_stack_load[i_t])
X = np.asarray(X)
y = np.asarray(y)

##################
# CLASSIFICATION #
##################

cv = 5
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
outer_results = list(); outer_scores = list()
for train_ix, test_ix in cv_outer.split(X):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    outer_results.append(clf.fit(X_train, y_train))
    outer_scores.append(clf.score(X_test, y_test))

#################
# PRINT RESULTS #
#################
for i_split, results in enumerate(outer_results):
    coefs_abs = np.abs(results.coef_)
    IX_coefs = np.argsort(-coefs_abs)
    coef_abs_sorted = coefs_abs[IX_coefs]
    print('Split ', i_split+1)
    print(results.alpha_)
    print(1+IX_coefs[:10])
    print(coef_abs_sorted[:10])

print(outer_scores)
print(np.mean(outer_scores))
