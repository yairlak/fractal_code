#!/usr/bin/env python
import sys, os, argparse, pickle, json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import decomposition
from sklearn import preprocessing
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Ablate and eval model on test dataset')
parser.add_argument('--model', type=str, default='../data/models/model_LSTM_p1_0.5_p2_0.5_nwords_1000000_length_clip_21_depth_clip_5_seed1_LSTM_dropout_0.1_nlayers_2_nhid_32_bptt_length_32_embsz_8.pt', help='path to trained pt model. If empty list then best model is taken based on p1 and p2.')
parser.add_argument('--sentences', type=str, default=[], help='Path to text file containing the list of sentences to analyze')
parser.add_argument('--meta', type=str, default=[], help='The corresponding meta data of the sentences')
parser.add_argument('--activations', default='../results/activations/model_LSTM_p1_0.5_p2_0.5_nwords_1000000_length_clip_21_depth_clip_5_seed1_LSTM_dropout_0.1_nlayers_2_nhid_32_bptt_length_32_embsz_8.pt_test_depth_3_length_2.txt.pkl' ,type=str, help='The corresponding sentence (LSTM) activations')
parser.add_argument('--p1', type=float, default=0.5)
parser.add_argument('--p2', type=float, default=0.5)
parser.add_argument('--depth', default=3, type=int)
parser.add_argument('--length', default=2, type=int)
parser.add_argument('--var-type', type=str, default='hidden', choices=['hidden', 'cell'])
parser.add_argument('--unit-to-plot', default=58, type=int)
parser.add_argument('--relative-pert', action='store_true', default=True)
parser.add_argument('--pc', type=int, default=1, help='If args.unit=0 (pertubation in pc direction), then this arg determines which PC to take (counting from 1!).')
parser.add_argument('--currents', type=float, default=[-10, -5, 0, 1, 10])
parser.add_argument('--t-for-activation-plot', default=[2, 4, 6], type=int)
args = parser.parse_args()

# Generate conditions
lw = 1
conditions = [] # list of conditions, in which each item is a list of tuples, each tuple is key-value pair from info
lss = []
colors = []
lws = []
for binary_ntuple in list(itertools.product([0, 1], repeat=3)):
    cond = [('number_%i'%(i+1), 'singular') if v==0 else ('number_%i'%(i+1), 'plural') for (i,v) in enumerate(binary_ntuple)]
    conditions.append(cond)
    if cond[0][1] == 'singular':
        colors.append('r')
    else:
        colors.append('b')


    if len(cond)>1: # if depth>=2
        if cond[1][1] == 'singular':
            lss.append('--')
        else:
            lss.append('-')

        if len(cond)>2: # depth>=3
            if cond[2][1] == 'singular':
                lws.append(3*lw)
            else:
                lws.append(lw)
        else:
            lws.append(lw)
    else:
        lss.append('-')
        lws.append(lw)

graphs = []
for cond, color, ls, lw in zip(conditions, colors, lss, lws):
    cond_str = ''
    for (key, value) in cond:
        cond_str += ' %s %s' % (key, value)
    graphs.append([1, color, ls, lw, args.unit_to_plot, 'hidden', cond_str])
# graphs = [[1, 'b', '--', 3, 23, 'hidden', 'number_1', 'singular']]


def get_unit_gate_and_indices_for_current_graph(graph, info, condition):
    '''

    :param graph: list containing info regarding the graph to plot - unit number, gate and constraints
    :param info: info objcet in Theo's format
    :param condition: sentence type (e.g., objrel, nounpp).
    :return:
    unit - unit number
    gate - gate type (gates.in, gates.forget, gates.out, gates.c_tilde, hidden, cell)
    IX_sentences - list of indices pointing to sentences numbers that meet the constraints specified in the arg graph
    '''
    color = graph[1]
    ls = graph[2]
    ls = ls.replace("\\", '')
    lw = float(graph[3])
    unit = int(graph[4])
    gate = graph[5]
    #print(len(graph))
    if len(graph) % 2 == 1:
        label = graph[-1]
    else:
        label = None
    # constraints are given in pairs such as 'number_1', 'singular' after unit number and gate
    kvs = graph[-1].strip().split(' ')
    keys_values = [(kvs[i], kvs[i + 1]) for i in range(0, len(kvs), 2)]
    if not label:
        label = str(unit) + '_' + gate + '_' + '_'.join([key + '_' + value for (key, value) in keys_values])
        label = ''.join([value[0].upper() for (key, value) in keys_values])
        # if args.use_tex:
        #     label = label.replace("_", "-")
    IX_to_sentences = []
    for i, curr_info in enumerate(info):
        check_if_contraints_are_met = True
        for key_value in keys_values:
            key, value = key_value
            if curr_info[key] != value:
                check_if_contraints_are_met = False
        if check_if_contraints_are_met:# and curr_info['RC_type']==condition:
            IX_to_sentences.append(i)
    return unit, gate, IX_to_sentences, label, color, ls, lw


def add_graph_to_plot(ax, LSTM_activations, unit, gate, label, c, ls, lw):
    '''

    :param LSTM_activations(ndarray):  LSTM activations for current *gate*
    :param stimuli (list): sentences over which activations are averaged
    :param unit (int): unit number to plot
    :param gate (str): gate type (gates.in, gates.forget, gates.out, gates.c_tilde, hidden, cell)
    :return: None (only append curve on active figure)
    '''
    #if LSTM_activations: print('Unit ' + str(unit))

    if gate.find('gate')==0: gate = gate[6::] # for legend, omit 'gates.' prefix in e.g. 'gates.forget'
    # Calc mean and std
    # if LSTM_activations:
    mean_activity = np.mean(LSTM_activations, axis=0)
    std_activity = np.std(LSTM_activations, axis=0)

    # Add curve to plot
    ax.errorbar(range(mean_activity.shape[0]), mean_activity, yerr=std_activity,
            label=label, ls=ls, lw=lw, color=c)

#####################################################
# Load stuff
#####################################################
# Model filename:
# Input sentences for evaluation:
if not args.sentences:
    args.sentences = '../data/datasets/test_depth_%i_length_%i.txt' % (args.depth, args.length)
    args.meta = '../data/datasets/test_depth_%i_length_%i.info' % (args.depth, args.length)

# Load meta data and generate indexing for each verb, whether it is singular or plural
# meta = pickle.load(open(args.meta, 'rb'))
with open(args.meta, "rb") as info_file:
    info = json.load(info_file)

with open(args.sentences, 'r') as f_sentences:
    sentences = f_sentences.readlines()

IX2sentences = {}
for v in range(args.depth):
    IX2sentences[v+1] = {}
    IX2sentences[v + 1]['singular'] = [i for i, s in enumerate(info) if s['number_%i' % (v + 1)] == 'singular']
    IX2sentences[v + 1]['plural'] = [i for i, s in enumerate(info) if s['number_%i' % (v + 1)] == 'plural']


fig_im, ax = plt.subplots(figsize=(20, 10))
####################
# PLOT ACTIVATIONS #
####################

with open(args.activations, 'rb') as f:
    LSTM_activations = pickle.load(f)[args.var_type]

X = []
for i_trial, trial_data in enumerate(LSTM_activations):
    trial_vector_t = []
    for i_t, t_data in enumerate(trial_data):
        trial_vector_t.append(t_data.flatten())
    X.append(trial_vector_t)
X = np.asarray(X).swapaxes(1, 2)


for g, graph in enumerate(graphs):
    unit, gate, IX_to_sentences, label, color, ls, lw = get_unit_gate_and_indices_for_current_graph(graph, info, '')
    graph_activations = X[IX_to_sentences, unit-1, :]
    curr_stimuli = [sentence for ind, sentence in enumerate(sentences) if ind in IX_to_sentences]
    add_graph_to_plot(ax, graph_activations, unit, gate, label, color, ls, lw)

    words = sentences[0].split(' ')
    words = [w[0] for w in words]
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words)

plt.tight_layout()
plt.savefig('../figures/unit_activations/unit_dynamics_%i_%s.png'%(args.unit_to_plot, args.var_type))
# plt.show()