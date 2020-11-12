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
parser.add_argument('--p1', type=float, default=0.5)
parser.add_argument('--p2', type=float, default=0.5)
parser.add_argument('--depth', default=3, type=int)
parser.add_argument('--length', default=2, type=int)
parser.add_argument('--unit', default=0, type=int)
parser.add_argument('--unit-to-plot', default=23, type=int)
parser.add_argument('--relative-pert', action='store_true', default=True)
parser.add_argument('--pc', type=int, default=2, help='If args.unit=0 (pertubation in pc direction), then this arg determines which PC to take (counting from 1!).')
parser.add_argument('--currents', type=float, default=[-10, -5, 0, 5, 10])
parser.add_argument('--t-for-activation-plot', default=[6,9], type=int)
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
    ax.errorbar(range(1, mean_activity.shape[0] + 1), mean_activity, yerr=std_activity,
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

#####################################################
# Load full model performance
#####################################################
filename_full_model = '%s_%s.acc' % (os.path.basename(args.model), os.path.basename(args.sentences))
filename_full_model = os.path.join('../results/model_eval', filename_full_model)
with open(filename_full_model, 'r') as f:
    performance = f.readlines()
performance = [s.strip('\n').split('\t')[1::] for s in performance]
performance = np.asarray(performance, dtype=int)
performance = np.flip(performance, axis=1) # order of verbs in ablation is from inner to outer, whereas v_1 is considered as the outmost verb!

performance_full_model = np.empty((3, args.depth)) # 1st row: all sentences; 2nd: singular cases; 3rd: plural cases
performance_full_model[:] = np.nan
for v in range(args.depth): # V_1 refers to the outmost verb
    performance_all_curr_verb = np.average(performance[:, v], axis=0)
    performance_singular_curr_verb = np.average(performance[IX2sentences[v + 1]['singular'], v], axis=0)
    performance_plural_curr_verb = np.average(performance[IX2sentences[v + 1]['plural'], v], axis=0)
    performance_full_model[0, v] = performance_all_curr_verb
    performance_full_model[1, v] = performance_singular_curr_verb
    performance_full_model[2, v] = performance_plural_curr_verb


###########################
# PCA non-perturbed model #
###########################
if args.unit == 0:  # PCA case
    # Reshape: concatenate all timepoints before PCAing (num_trials*num_timepoints X num_features)
    print('Loading activations...')
    activations = '../results/activations/%s_%s.pkl' % (os.path.basename(args.model), os.path.basename(args.sentences))
    LSTM_activations_no_perturb = pickle.load(open(activations, 'rb'))
    LSTM_activations_no_perturb = LSTM_activations_no_perturb['hidden']

    print('Prepare data for PCA: breaking down sentences into words')
    design_matrix_all_words = np.vstack(
        [np.reshape(word_data, (1, -1), order='C') for sentence_data in tqdm(LSTM_activations_no_perturb) for word_data in
         sentence_data])

    print('Run PCA')
    pca = decomposition.PCA(n_components=2)
    pca.fit(design_matrix_all_words)

#########################################################
# Collect performance after injecting current into unit #
#########################################################
num_subplots = len(args.currents)
num_timepoint_to_plot = len(args.t_for_activation_plot)
num_words = len(sentences[0].split(' '))

# perf = np.zeros((3, args.depth, num_words)) # 3 for sing/plur/both
performance_ablated_model = np.empty((3, args.depth, num_words))

fig_im, axs = plt.subplots(num_subplots, num_timepoint_to_plot+3, figsize=(10 + num_timepoint_to_plot*5, 4 * num_subplots))
for i_ax, current in enumerate(args.currents):
    for inject_time in range(num_words):
        filename = '%s_%s_injected_to_%s_unit_%i_current_%1.2f_inject_time_%s' % (os.path.basename(args.model), os.path.basename(args.sentences), 'hidden', args.unit, current, inject_time)
        if args.unit == 0: # PC case
            filename += '_pc_%i' % args.pc
        if args.relative_pert:
            filename += '_relative_pert'
        filename = os.path.join('../results/model_eval', filename + '.acc')

        print('Loading: %s' % filename)
        with open(filename, 'r') as f:
            performance = f.readlines()
        performance = [s.strip('\n').split('\t')[1::] for s in performance]
        performance = np.asarray(performance, dtype=int)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
        # order of verbs in ablation is from inner to outer, whereas v_1 is considered as the outmost verb!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
        performance = np.flip(performance, axis=1)

        for v in range(args.depth):  # V_1 refers to the outmost verb
            performance_ablated_model[0, v, inject_time] = np.average(performance[:, v], axis=0) # Both
            performance_ablated_model[1, v, inject_time] = np.average(performance[IX2sentences[v + 1]['singular'], v], axis=0) # singular
            performance_ablated_model[2, v, inject_time] = np.average(performance[IX2sentences[v + 1]['plural'], v], axis=0) # plural


        ####################
        # PLOT ACTIVATIONS #
        ####################

        if inject_time in args.t_for_activation_plot:
            IX_t = args.t_for_activation_plot.index(inject_time)
            with open(filename + '.pkl', 'rb') as f:
                LSTM_activations = pickle.load(f)

            if args.unit == 0: # PCA case. Project activations with pca projection of non-perturbed network
                print('Prepare data for PCA: breaking down sentences into words')
                design_matrix_all_words = np.vstack([np.reshape(word_data, (1, -1), order='C') for sentence_data in tqdm(LSTM_activations) for word_data in sentence_data])
                words_PCA_projected = pca.transform(design_matrix_all_words)

                PCA_trajectories = []
                st = 0
                for _, sentence_data in enumerate(tqdm(LSTM_activations)):
                    curr_sentence_data = []
                    for _ in sentence_data:
                        curr_sentence_data.append(words_PCA_projected[st, :])
                        st += 1
                    PCA_trajectories.append(np.asarray(
                        curr_sentence_data).transpose())  # list (len=num_trials) of ndarrays (num_timepoints X num_PC_components)
                PCA_trajectories = np.asarray(PCA_trajectories)

            for g, graph in enumerate(graphs):
                unit, gate, IX_to_sentences, label, color, ls, lw = get_unit_gate_and_indices_for_current_graph(graph, info, '')
                if args.unit > 0:
                    graph_activations = LSTM_activations[0, IX_to_sentences, unit-1, :]
                elif args.unit == 0:
                    graph_activations = PCA_trajectories[IX_to_sentences, args.pc-1, :]
                curr_stimuli = [sentence for ind, sentence in enumerate(sentences) if ind in IX_to_sentences]
                add_graph_to_plot(axs[i_ax, IX_t], graph_activations, unit, gate, label, color, ls, lw)

    words = sentences[0].split(' ')
    words = [w[0] for w in words]
    for i_spb, spb in enumerate(['Both', 'Singular', 'Plural']):
        im = axs[i_ax, -3+i_spb].imshow(performance_ablated_model[i_spb, :, :], origin='lower', vmin=0, vmax=1, cmap='RdBu_r')

        if i_ax == num_subplots-1:
            axs[i_ax, -3+i_spb].set_xlabel(spb, fontsize=16)
        # axs[i_ax, -1].set_ylabel('Verb', fontsize=16)
        axs[i_ax, -3+i_spb].set_xticks(range(len(words)))
        axs[i_ax, -3+i_spb].set_xticklabels(words)
        axs[i_ax, -3+i_spb].set_yticks(range(3))
        axs[i_ax, -3+i_spb].set_yticklabels(['V1', 'V2', 'V3'])
        axs[i_ax, -3+i_spb].set_title('Injection value = %1.2f' % current)

    for j, t_inject in enumerate(args.t_for_activation_plot):
        axs[i_ax, j].set_xticks(range(1, len(words)+1))
        axs[i_ax, j].set_xticklabels(words)
        axs[i_ax, j].get_xticklabels()[t_inject].set_color('red')
        # axs[i_ax, j].set_yticks([-1, 0, 1])
        # axs[i_ax, 0].set_ylim([-1, 1])
        # axs[i_ax, 0].set_yticklabels(['V1', 'V2', 'V3'])

        if args.unit > 0:
            axs[i_ax, j].set_title(
                'Unit %i, injection time-point %i' % (args.unit_to_plot, args.t_for_activation_plot[j]+1))
            axs[i_ax, j].set_ylabel('h_t')
        else:
            if i_ax == 0:
                axs[i_ax, j].set_title(
                    'Injection time-point %i' % (args.t_for_activation_plot[j]+1))
            if j == 0:
                axs[i_ax, j].set_ylabel('PC-%i' % args.pc)
            else:
                axs[i_ax, j].set_ylabel('')

plt.tight_layout()
# fig_im.subplots_adjust(right=0.8, hspace=0.7)
fig_im.subplots_adjust(right=0.9)
cbar_ax = fig_im.add_axes([0.95, 0.15, 0.01, 0.7])
fig_im.colorbar(im, cax=cbar_ax)
cbar_ax.get_yaxis().labelpad = 15
cbar_ax.set_ylabel('Accuracy', fontsize=16, rotation=270)

plt.savefig('noise_exp_%s.png'%args.unit)
plt.show()