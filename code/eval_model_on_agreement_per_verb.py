#!/usr/bin/env python
import os, pickle
import argparse
from tqdm import tqdm
import sys
sys.path.append('../data')
import common
import numpy as np

parser = argparse.ArgumentParser(description='Evaluate pt model on an agreement task')
parser.add_argument('--model', type=str, default='../data/models/model_LSTM_p1_0.5_p2_0.5_nwords_1000000_length_clip_21_depth_clip_5_seed1_LSTM_dropout_0.1_nlayers_2_nhid_32_bptt_length_32_embsz_8.pt', help='Meta file stored once finished training the corpus')
parser.add_argument('--sentences', default=[], type=str, help='Path to text file containing the list of sentences to analyze')
parser.add_argument('--meta', default=[], type=str, help='The corresponding meta data of the sentences')
parser.add_argument('--activations', default=[] ,type=str, help='The corresponding sentence (LSTM) activations')
parser.add_argument('--p1', type=float, default=0.5)
parser.add_argument('--p2', type=float, default=0.5)
parser.add_argument('--num_units', default=[], type=int, help='if model is not given then this will be determined from the best_model from the grid search')
parser.add_argument('--depth', default=3, type=int)
parser.add_argument('--length', default=2, type=int)
parser.add_argument('--vocabulary', default='', type=str, help='path to vocabulary. If empty then vocab extension is added to the filename of the model')
parser.add_argument('--noun-initial', default='n', help='The character with which noun tokens begin. Results will include agreement performance on all verbs')
parser.add_argument('--verb-initial', default='v', help='The character with which verb tokens begin. Results will include agreement performance on all verbs')
parser.add_argument('--feature-values', default=['sg', 'pl'], help='(list of of str) The corresponding feature values in the input and results vectors')
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--output', default='../results/model_eval/', type=str ,help='path to results folder')
args = parser.parse_args()


if not args.sentences:
    args.sentences = '../data/datasets/test_depth_%i_length_%i.txt' % (args.depth, args.length)
    args.meta = '../data/datasets/test_depth_%i_length_%i.info' % (args.depth, args.length)

if not args.activations:
    args.activations = '../results/activations/%s_%s.pkl' % (os.path.basename(args.model), os.path.basename(args.sentences))

os.makedirs(os.path.join(args.output), exist_ok=True)
filename = os.path.basename(args.model) + '_' + os.path.basename(args.sentences) + '.acc'
filename = os.path.join(args.output, filename)

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



# ----------------------
# --- Load text data ---
# ----------------------
sentences_str = [l.rstrip('\n').split(' ') for l in open(args.sentences, encoding='utf-8')]
if args.vocabulary:
    path2vocab = args.vocabulary
else:
    path2vocab = args.model + '.vocab'
vocab = common.Dictionary(path2vocab)


singular_verb_IXs_in_output_layer, plural_verb_IXs_in_output_layer= get_verb_indexes_in_onehot_encoding(vocab, args.verb_initial, args.feature_values)
print('vocabulary: ' + ' '.join([str(v)+':'+str(k) for k, v in vocab.word2idx.items()]))
print('indexes to singluar forms of the verb in output layer: ', singular_verb_IXs_in_output_layer)
print('indexes to plural forms of the verb in output layer: ', plural_verb_IXs_in_output_layer)

# ----------------------------------------
# --- Load unit and output activations ---
# ----------------------------------------
print('Loading activations...')
LSTM_activations = pickle.load(open(args.activations, 'rb'))

# ------------------------------------------------------------------------------
# ------Get and compare probabilities of target and wrong verbs for each sentence
# ------------------------------------------------------------------------------
scores = []
for i, sentence_str in enumerate(tqdm(sentences_str)):
    out = LSTM_activations['log_probabilities_all_words'][i]
    scores_on_sentence = []
    _, _, verb_positions = get_agreements_center_embedding(sentence_str, args.noun_initial, args.verb_initial)

    for j in [v_pos - 1 for v_pos in verb_positions]: # check if at one step before the verb
        correct_word = sentence_str[j+1] # Correct word prediction from ground-truth
        correct_feature = correct_word[1+correct_word.find('['):correct_word.find(']')].strip()
        if correct_feature == 'sg':
            log_ps_for_correct_feature = [out[j, v] for v in singular_verb_IXs_in_output_layer]
            log_ps_for_wrong_feature = [out[j, v] for v in plural_verb_IXs_in_output_layer]
        elif correct_feature == 'pl':
            log_ps_for_correct_feature = [out[j, v] for v in plural_verb_IXs_in_output_layer]
            log_ps_for_wrong_feature = [out[j, v] for v in singular_verb_IXs_in_output_layer]
        scores_on_sentence.append(int(np.mean(log_ps_for_correct_feature)>np.mean(log_ps_for_wrong_feature)))

    scores.append(scores_on_sentence)
# ----------------------------
# --- save results to file ---
# ----------------------------
with open(filename, 'w') as f:
    for sentence_str, scores_on_sentence in zip(sentences_str, scores):
        curr_line = '%s\t' % (' '.join(sentence_str)) + '\t'.join(map(str, scores_on_sentence)) + '\n'
        f.write(curr_line)
print('accuracy results saved to: %s' % filename)  #os.path.join(args.output, args.model_name, args.data_type, filename) )
