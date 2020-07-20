#!/usr/bin/env python
import sys, os, argparse
import argparse
import numpy as np
import common, utils
import copy
import torch
from torch.autograd import Variable
import pickle
from pprint import pprint

parser = argparse.ArgumentParser(description='Ablate and eval model on test dataset')
parser.add_argument('--model', type=str, default='../data/models/model_LSTM_p1_0.5_p2_0.5_nwords_1000000_length_clip_21_depth_clip_5_seed1_LSTM_dropout_0.1_nlayers_2_nhid_32_bptt_length_32_embsz_8.pt', help='path to trained pt model. If empty list then best model is taken based on p1 and p2.')
parser.add_argument('--sentences', type=str, default=[], help='Path to text file containing the list of sentences to analyze')
parser.add_argument('--meta', type=str, default='../../datasets/artificial/test_depth_2_length_2.info', help='The corresponding meta data of the sentences')
parser.add_argument('--p1', type=float, default=0.5)
parser.add_argument('--p2', type=float, default=0.5)
parser.add_argument('--depth', default=3, type=int)
parser.add_argument('--length', default=2, type=int)
parser.add_argument('--vocabulary', default='', type=str, help='path to vocabulary. If empty then vocab extension is added to the filename of the model')
parser.add_argument('--noun-initial', default='n', help='The character with which noun tokens begin. Results will include agreement performance on all verbs')
parser.add_argument('--verb-initial', default='v', help='The character with which verb tokens begin. Results will include agreement performance on all verbs')
parser.add_argument('--feature-values', default=['sg', 'pl'], help='(list of of str) The corresponding feature values in the input and results vectors')
parser.add_argument('--activation-type', default='hidden',type=str, help='hidden/cell')
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--relative-pert', action='store_true', default=True)
parser.add_argument('--unit', default=0, type=int, help='List of test units to inject - counting starts at 1 (not zero). If zero, inject in pca direction')
parser.add_argument('--path2pca', default=[], type=str ,help='Path the pca results, with the 2PCs along which noise would be injected (if args.unit=0)')
parser.add_argument('--pc', type=int, default=2, help='If args.unit=0 (pertubation in pc direction), then this arg determines which PC to take (counting from 1!).')
parser.add_argument('--current', type=float, default=-10)
parser.add_argument('--inject-time', type=int, action='append', default=[])
parser.add_argument('--output', default='../results/model_eval/', type=str ,help='path to results folder')
args = parser.parse_args()



#######################################
###### LOAD MODEL & INPUTS   ##########
#######################################

# Input sentences for evaluation:
if not args.sentences:
    args.sentences = '../data/datasets/test_depth_%i_length_%i.txt' % (args.depth, args.length)
    args.meta = '../data/datasets/test_depth_%i_length_%i.info' % (args.depth, args.length)


os.makedirs(os.path.join(args.output), exist_ok=True)

if not args.vocabulary:
    args.vocabulary = args.model + '.vocab'

# Vocabulary
vocab = common.Dictionary(args.vocabulary)

# Sentences
sentences = [l.rstrip('\n').split(' ') for l in open(args.sentences, encoding='utf-8')]
singular_verb_IXs_in_output_layer, plural_verb_IXs_in_output_layer= utils.get_verb_indexes_in_onehot_encoding(vocab, args.verb_initial, args.feature_values)
print(vocab.word2idx)
print(singular_verb_IXs_in_output_layer, plural_verb_IXs_in_output_layer)

# Load model
model = torch.load(args.model, map_location=lambda storage, loc: storage)  # requires GPU model
model.rnn.flatten_parameters()
model_orig_state = copy.deepcopy(model.state_dict())

# Load pca results
if not args.path2pca:
    path2pca = '../results/activations/pca/'
    fn = 'pca_1d_%s_depth_%i_length_%i_%s' % (args.activation_type, args.depth, args.length, os.path.basename(args.model))
    args.path2pca = os.path.join(path2pca, fn+'.csv')
PCs = np.genfromtxt(args.path2pca, delimiter=',') # num_PCs * network_size

pprint(args)
########################################
###### INJECT AND EVAL MODEL  ##########
########################################

if args.cuda:
    model = model.cuda()

# To know the activity of each units to change in the tensor, construct a logical index array, compatible with both args.unit=0 (pca) or args.unit>0 (single unit)
pertubation = torch.zeros([model.nlayers, 1, model.nhid])
if args.unit > 0:
    layer_of_unit = int(np.floor(args.unit/model.nhid)) # counting from zero
    unit_serial_num_in_layer = (args.unit-1) % model.nhid # args.unit is counted from one whereas unit_serial_num_in_layer from zero
    pertubation[layer_of_unit, 0, unit_serial_num_in_layer] = 1
elif args.unit == 0: # inject in PCA direction
    # PCs = np.flip(PCs,axis=1)
    for l in range(model.nlayers):
        st = l * model.nhid
        ed = (l+1) * model.nhid
        pertubation[l, 0, :] = torch.from_numpy(PCs[args.pc-1, st:ed])


scores = []
activations = {}
if args.activation_type == 'hidden':
    hc = 0
elif args.activation_type == 'cell':
    hc = 1
else:
    raise ()
filename = '%s_%s_injected_to_%s_unit_%i_current_%1.2f_inject_time_%s' % (
os.path.basename(args.model), os.path.basename(args.sentences), args.activation_type, args.unit, args.current,
'_'.join(map(str, args.inject_time)))
if args.unit == 0:
    filename += '_pc_' + str(args.pc)
if args.relative_pert:
    filename += '_relative_pert'
filename = os.path.join(args.output, filename + '.acc')

activations[args.activation_type] = []
for i, sentence_str in enumerate(sentences):
    activations_curr_sentence = []
    scores_on_sentence = []
    _, _, verb_positions = utils.get_agreements_center_embedding(sentence_str, args.noun_initial, args.verb_initial)

    # Init model
    hidden = model.init_hidden(1)
    inp = Variable(torch.LongTensor([[vocab.word2idx["<eos>"]]]))
    if args.cuda:
        inp = inp.cuda()
    out, hidden = model(inp, hidden)
    # activations_curr_sentence.append(hidden[0].detach().numpy().flatten())

    # present sentence
    for j, w_str in enumerate(sentence_str):
        # print(w_str)
        inp = Variable(torch.LongTensor([[vocab.word2idx[w_str]]]))
        if args.cuda:
            inp = inp.cuda()

        out_pertubed, _ = model.forward_pertubation(inp, hidden, pertubation * args.current)
        out, hidden = model(inp, hidden)

        # Add perturbation
        if j in args.inject_time:
            curr_h_value = hidden[hc].data  # [layer_of_unit, 0, unit_serial_num_in_layer]
            if args.relative_pert:
                # NOTE: hidden var includes both hidden and cell states. The first [0] is the hidden:
                hidden[hc].data = curr_h_value + pertubation * args.current  # curr_h_value + perturbation * args.current
            else:
                hidden[hc].data = pertubation * args.current
            out = out_pertubed


        activations_curr_sentence.append(hidden[hc].detach().numpy())
        out = torch.nn.functional.log_softmax(out, dim=-1).unsqueeze(0)


        if j in [v_pos - 1 for v_pos in verb_positions]:  # check if at one step before the verb
            correct_word = sentence_str[j + 1]  # Correct word prediction from ground-truth
            correct_feature = correct_word[1 + correct_word.find('['):correct_word.find(']')].strip()
            if correct_feature == 'sg':
                log_ps_for_correct_feature = [out[0, 0, 0, v].data.item() for v in singular_verb_IXs_in_output_layer]
                log_ps_for_wrong_feature = [out[0, 0, 0, v].data.item() for v in plural_verb_IXs_in_output_layer]
            elif correct_feature == 'pl':
                log_ps_for_correct_feature = [out[0, 0, 0, v].data.item() for v in plural_verb_IXs_in_output_layer]
                log_ps_for_wrong_feature = [out[0, 0, 0, v].data.item() for v in singular_verb_IXs_in_output_layer]

            scores_on_sentence.append(int(np.mean(log_ps_for_correct_feature) > np.mean(log_ps_for_wrong_feature)))
            # print(scores_on_sentence[-1])

    # activations_curr_sentence = np.stack(activations_curr_sentence, axis=-1) # stack on a new dim (last)
    # Shape of activations_curr_sentence = (num_units, num_timepoints) # axis=1 is for the batch, which in this case is one
    activations[args.activation_type].append(activations_curr_sentence)
    scores.append(scores_on_sentence)
# ----------------------------
# --- save results to file ---
# ----------------------------
with open(filename, 'w') as f:
    for i, (sentence_str, scores_on_sentence) in enumerate(zip(sentences, scores)):
        curr_line = '%s\t' % (' '.join(sentence_str)) + '\t'.join(map(str, scores_on_sentence)) + '\n'
        f.write(curr_line)
print('Saved to %s: ' % filename)

activations_all_sentences = np.stack(activations[args.activation_type], axis=0) # axis=0 will generate array: num_trials X num_units X num_timepoints
activations_mean = np.mean(activations_all_sentences, axis=1)
activations_std = np.std(activations_all_sentences, axis=1)
with open(filename + '.pkl', 'wb') as f:
     # pickle.dump([activations_mean, activations_std], f)
     pickle.dump(activations_all_sentences, f)
