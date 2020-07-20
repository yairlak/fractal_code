#!/usr/bin/env python
import sys
import math
import os
import torch
import argparse 
sys.path.append('..') # To load the model, models/rnn/ (models.rnn) should be in path
import numpy as np
import pickle
from tqdm import tqdm
import lstm

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--model', type=str, default='../data/models/model_LSTM_p1_0.5_p2_0.5_nwords_1000000_length_clip_21_depth_clip_5_seed1_LSTM_dropout_0.1_nlayers_2_nhid_32_bptt_length_32_embsz_8.pt', help='Meta file stored once finished training the corpus')
parser.add_argument('--p1', type=float, default=0.5)
parser.add_argument('--p2', type=float, default=0.5)
parser.add_argument('--depth', type=int, default=3)
parser.add_argument('--length', type=int, default=2)
parser.add_argument('--input', default=[], help='Input sentences. If not given set based on length and depth args')
parser.add_argument('--vocabulary', default=[])
parser.add_argument('--path2output', default='../results/activations/', help="'../../results/model_activations/$MODEL-NAME/$DATA-TYPE/', output path for the output vectors'")
parser.add_argument('--logit-only', action='store_true', default=False)
parser.add_argument('--eos-separator', default='<eos>')
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--overwrite', action='store_true', default=True)
args = parser.parse_args()

if not args.input:
    args.input = '../data/datasets/test_depth_%i_length_%i.txt' % (args.depth, args.length)

elif len(args.get_representations) == 0:
    raise RuntimeError("Please specify at least one of -g lstm or -g word")

if not args.vocabulary:
    args.vocabulary = os.path.join(args.model + '.vocab')

os.makedirs(args.path2output, exist_ok=True)
fn_activations = os.path.join(args.path2output, '%s_%s.pkl' % (os.path.basename(args.model), os.path.basename(args.input)))
if os.path.isfile(fn_activations) and (not args.overwrite):
    print('filename already exists: %s' % fn_activations)
    exit()

class Dictionary(object):
    def __init__(self, path=None):
        self.word2idx = {}
        self.idx2word = []
        if path:
            self.load(path)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def load(self, path):
        with open(path, 'r') as f:
            for line in f:
                self.add_word(line.rstrip('\n'))

    def save(self, path):
        with open(path, 'w') as f:
            for w in self.idx2word:
                f.write('{}\n'.format(w))

vocab = Dictionary(args.vocabulary)
sentences = []
print(args.input)
for l in open(args.input):
    sentence = l.rstrip('\n').split(" ") 
    sentences.append(sentence)
sentences = np.array(sentences)


print('Loading models...', file=sys.stderr)
sentence_length = [len(s) for s in sentences]
max_length = max(*sentence_length)
model = torch.load(args.model, map_location=lambda storage, loc: storage)

if args.cuda:
    model.cuda()
    model.device = 'cuda:0'
    print('Model is on Cuda %s' % next(model.parameters()).is_cuda)
else:
    model.device = 'cpu'
print(model)

# hack the forward function to send an extra argument containing the model parameters
if not args.logit_only: #'lstm' in args.get_representations:
    model.rnn.forward = lambda input, hidden: lstm.forward(model.rnn, input, hidden)

saved = {}

def feed_sentence(model, h, sentence):
    outs = []
    for w in sentence:
        out, h = feed_input(model, h, w)
        outs.append(torch.nn.functional.log_softmax(out[0]).unsqueeze(0))
    return outs, h

def feed_input(model, hidden, w):
    inp = torch.autograd.Variable(torch.LongTensor([[vocab.word2idx[w]]]))
    if args.cuda:
        inp = inp.cuda()
    out, hidden = model(inp, hidden)

    return out, hidden

# init arrays
print('Extracting LSTM representations', file=sys.stderr)
log_probabilities = [np.zeros(len(s)) for s in tqdm(sentences)] # np.zeros((len(sentences), max_length))
log_probabilities_all_words = [np.zeros((len(s), len(vocab.word2idx.keys()))) for s in tqdm(sentences)] # np.zeros((len(sentences), max_length))
if not args.logit_only:
    vectors = {k: [np.zeros((model.nhid*model.nlayers, len(s))) for s in sentences] for k in tqdm(['gates.in', 'gates.forget', 'gates.out', 'gates.c_tilde'])} #np.zeros((len(sentences), model.nhid*model.nlayers, max_length)) for k in ['in', 'forget', 'out', 'c_tilde']}
    vectors['hidden'] = []
    vectors['cell'] = []
# init model
hidden = model.init_hidden(bsz=1)
init_sentence = args.eos_separator
init_sentence = [s.lower() for s in init_sentence.split(" ")]
init_out, init_h = feed_sentence(model, hidden, init_sentence)

for i, s in enumerate(tqdm(sentences)):
    #sys.stdout.write("{}% complete ({} / {})\r".format(int(i/len(sentences) * 100), i, len(sentences)))
    out = init_out[-1]
    hidden = init_h
    sentence_activations_hidden, sentence_activations_cell = ([], [])
    for j, w in enumerate(s):
        # store the surprisal for the current word
        log_probabilities[i][j] = out[0,0,vocab.word2idx[w]].data.item()
        log_probabilities_all_words[i][j] = out[0,0,:].data.numpy()
        inp = torch.autograd.Variable(torch.LongTensor([[vocab.word2idx[w]]]))
        if args.cuda:
            inp = inp.cuda()
        out, hidden = model(inp, hidden)
        out = torch.nn.functional.log_softmax(out[0]).unsqueeze(0)

        if not args.logit_only:
            sentence_activations_hidden.append(hidden[0].data.cpu().numpy())
            sentence_activations_cell.append(hidden[1].data.cpu().numpy())
            # we can retrieve the gates thanks to the hacked function
            for k, gates_k in vectors.items():
                if 'gates' in k:
                    k = k.split('.')[1]
                    gates_k[i][:,j] = torch.cat([g[k].data for g in model.rnn.last_gates],1).cpu().numpy()
        # save the results
        saved['log_probabilities']           = log_probabilities
        saved['log_probabilities_all_words'] = log_probabilities_all_words

        #if args.format != 'hdf5':
        saved['sentences'] = sentences

        saved['sentence_length'] = np.array(sentence_length)

    vectors['hidden'].append(sentence_activations_hidden)
    vectors['cell'].append(sentence_activations_cell)
    if not args.logit_only:
        for k, g in vectors.items():
            saved[k] = g

print ("Perplexity: {:.2f}".format(
    math.exp(
        sum(-lp.sum() for lp in log_probabilities)/
        sum((lp!=0).sum() for lp in log_probabilities))))

    
with open(fn_activations, 'wb') as fout:
    print(saved.keys())
    pickle.dump(saved, fout, -1)
print('Extracted activations were saved to: %s' % fout)
