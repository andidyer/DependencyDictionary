import numpy as np
import pandas as pd
import time
import re
import scipy
import json
from collections import defaultdict
from sys import stdin, stdout, stderr, argv
import argparse
import os

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 50)

def eprint(string):
    print(string,file=stderr)

argparser = argparse.ArgumentParser('For making a Universal Dependencies matrix')
argparser.add_argument('conllfile', help='the language\'s CONLLU file')
argparser.add_argument('outfile', help='the file that the matrix should be printed into')
argparser.add_argument('UDlabels',help='a list of possible UD labels')
argparser.add_argument('UDtags',help='a list of possible UD tags')
argparser.add_argument('-Pdist',help='Probability distribution over features and words.\n conditional = conditional probability; joint = joint probability', default='joint')
argparser.add_argument('-deprel_cxt', action='store_true',help='use deprel contexts; default True', default=False)
argparser.add_argument('-pos_cxt', action='store_true',help='use UPOS contexts; default True', default=False)
argparser.add_argument('-self_cxt', action='store_true',help='use token self context; default True', default=False)
argparser.add_argument('-parent_cxt', action='store_true',help='use token parent context; default True', default=False)
argparser.add_argument('-daughter_cxt', action='store_true',help='use token daughter context; default False', default=False)
argparser.add_argument('-threshold', help='Apply a frequency count minimum; default 5', type=int, default=5)
argparser.add_argument('-removeN', help='Remove top N most frequent words', type=int, default=0)

argparser.add_argument('-e','--epsilon', type=float, help='Apply a non-zero value to avoid NaN; default 1e-7', default=1e-7)
argparser.add_argument('--num_norm', action='store_true',help='Normalise number tokens to <num>; default False')
argparser.add_argument('--verbose', action='store_true',help='Print what it\'s doing at each point')
argparser.add_argument('--timer', action='store_true',help='Time how long the whole process takes')
args = argparser.parse_args()

if args.timer:
    start = time.time()

# Count vocabulary and create w2i defaultdict of tokens.
# Frequency threshold turns tokens with less than thr instances into UNK.
# Applies space and number normalisation.
def countVocab(conllfile, thr=0, removeN=0, num_norm=True):
    """counts vocab of input file and makes w2i"""
    _freq_ = {'<unk>':0}
    w2i = defaultdict(lambda: len(w2i))
    UNK = w2i['<unk>']

    num_token = re.compile('\d+')
    space = re.compile('\s+')

    with open(conllfile) as infile:
        tab_pattern = re.compile('^[^\t]+\t([^\t]+)\t')
        for line in infile:
            line = line.strip().lower()
            search = tab_pattern.search(line)
            if search:
                form = search.group(1)
                form = space.sub('',form)
                if num_norm:
                    form = num_token.sub('<num>',form)

                try:
                    _freq_[form] += 1
                except KeyError:
                    _freq_[form] = 1

    if removeN:
        remove_words = sorted(list(w for w in _freq_ if w!='<unk>'), key=lambda x: _freq_[x], reverse=True)[:removeN]
        for word in remove_words:
            del _freq_[word]
                    
    for w in (w for w in _freq_ if _freq_[w] > thr):
        _ = w2i[w]
    w2i = defaultdict(lambda: w2i['<unk>'], w2i)

    freq = {v:_freq_[k] for k,v in w2i.items()}
    return w2i, freq

# Get dependency contexts.
def getContexts(conll, num_norm = True):
    """Reads conll file and gets UD contexts.
    Is returned as generator"""

    root_dummy = [0,'**root**', '<ROOT>', 0, 'root']
    
    with open(conll, encoding='utf-8') as treebank:
        tabs = [root_dummy]

        pattern = re.compile('^(?:[^\t]+\t){9}[^\t]+$')
        fine_deprel = re.compile(':\w+$')
        num_token = re.compile('\d+')
        space = re.compile('\s+')

        for line in treebank:
            line = line.strip()
            if pattern.match(line):
                index, form, lemma, UPOS, XPOS, feats, head, deprel, deps, misc = line.split('\t')
                deprel = fine_deprel.sub('',deprel)
                form = space.sub('', form)
                if num_norm:
                    form = num_token.sub('<norm>',form)
                try:
                    tabs.append([int(index), form, UPOS, int(head), deprel])
                except ValueError:
                    assert re.match('\d+[-.]\d+', index), ('Unknown error not caused by range argument)')
                    pass #This is probably a range item

            if len(line)==0 or line[0]=='#':
                outdict = dict()
                for index,form,UPOS,head,deprel in tabs[1:]:
                    try:
                        _, hform, hPOS, _, hdep = tabs[head]
                    except:
                        raise Exception('Could not read.')
                    outdict['form'] = form
                    outdict['self_deprel'] = deprel
                    outdict['self_UPOS'] = UPOS
                    outdict['parent_deprel'] = hdep
                    outdict['parent_form'] = hform
                    outdict['parent_UPOS'] = hPOS
                            
                    yield outdict

                tabs = [root_dummy]

def UDcontextmatrix(deprel_context = True, pos_context = True, nrow=10000, parent_context=True, daughter_context=True, self_context=True):
    """Create matrix"""
    if not any([parent_context, daughter_context, self_context]):
        raise ValueError('Cannot make matrix without any self, parent or daughter contexts.')
    if not any([deprel_context, pos_context]):
        raise ValueError('Cannot make matrix without any syntax contexts.')
    d = {}
    if deprel_context:
        for l in UDlabels:
            if parent_context: d[l+'(p)'] = np.zeros(nrow).tolist()
            if daughter_context: d[l+'(d)'] = np.zeros(nrow).tolist()
            if self_context: d[l+'(s)'] = np.zeros(nrow).tolist()
    if pos_context:
        for t in UDtags:
            if t == '<ROOT>': continue
            if parent_context: d[t+'(p)'] = np.zeros(nrow).tolist()
            if daughter_context: d[t+'(d)'] = np.zeros(nrow).tolist()
            if self_context: d[t+'(s)'] = np.zeros(nrow).tolist()
    #print(json.dumps(d, indent=4))
    df = pd.DataFrame(data=d)
    return df

def populateMatrix(conllfile, matrix, num_norm=True,
                   in_place=True, epsilon=1e-3,
                   deprel_context = True, pos_context = True,
                   self_context=True,
                   parent_context=True,
                   daughter_context=True):

    if not in_place:
        matrix = matrix.copy()

    for word in getContexts(conllfile, num_norm=num_norm):
        _sf, _sd, _spos, _pd, _pf, _ppos = list(word[k] for k in ['form','self_deprel','self_UPOS',
                                                    'parent_deprel','parent_form','parent_UPOS'])

        if self_context:
            if deprel_context: matrix.loc[w2i[_sf], _sd+'(s)'] += 1 #self deprel
            if pos_context: matrix.loc[w2i[_sf], _spos+'(s)'] += 1 #self POS
        if parent_context:
            if deprel_context: matrix.loc[w2i[_sf], _pd+'(p)'] += 1 #parent deprel
            if _ppos == '<ROOT>': continue
            if pos_context: matrix.loc[w2i[_sf], _ppos+'(p)'] += 1 #parent POS
        if daughter_context:
            if _pf == '**root**' or _ppos == '<ROOT>': continue
            if deprel_context: matrix.loc[w2i[_pf], _sd+'(d)'] += 1 #daughter deprel
            if pos_context: matrix.loc[w2i[_pf], _spos+'(d)'] += 1 #daughter POS

    matrix = matrix+epsilon
            
    return matrix

UDlabels = list((label.strip() for label in open(args.UDlabels).readlines()))
UDtags = list((tag.strip() for tag in open(args.UDtags).readlines()))

conllfile = args.conllfile

    
#Hparams
epsilon = args.epsilon # a smoothing value
num_norm = args.num_norm
THR = args.threshold
removeN = args.removeN
deprel_context = args.deprel_cxt
pos_context = args.pos_cxt
self_context = args.self_cxt
parent_context = args.parent_cxt
daughter_context = args.daughter_cxt
P_dist = args.Pdist

if args.verbose:
    print('Reading CONLL file',file=stderr)
    print('Counting vocabulary',file=stderr)

w2i, freq = countVocab(conllfile, thr = THR, removeN = removeN, num_norm = num_norm)

if args.verbose:
    eprint('Vocabulary processed')
    eprint('Creating context matrix')

matrix = UDcontextmatrix(deprel_context, pos_context, nrow=len(w2i),
                         self_context=self_context,
                         parent_context=parent_context,
                         daughter_context=daughter_context)
if args.verbose:
    eprint('Matrix created')
    eprint('Populating matrix with values')

matrix = populateMatrix(conllfile, matrix, num_norm=num_norm,
               in_place=True, epsilon=epsilon,
                deprel_context=deprel_context,
                pos_context=pos_context,
                self_context=self_context,
               parent_context=parent_context,
               daughter_context=daughter_context)

if THR:
    matrix = matrix.iloc[1:,]

del freq #Frequency dictionary no longer needed

if args.verbose:
    eprint('Matrix populated')
    eprint('Converting to {} distribution'.format(P_dist))

def convert_to_P(matrix, P_dist = 'conditional', in_place=True):
    if not in_place:
        matrix = matrix.copy()
    word_count = matrix.values.sum()
    P_word = (matrix.sum(axis=1) / word_count)[:,None]
    matrix = matrix / matrix.sum(axis=1)[:,None]
    if P_dist == 'normal':
        norm = np.linalg.norm(matrix)
        return matrix / norm
    if P_dist == 'conditional':
        return matrix
    elif P_dist == 'joint':
        return matrix * P_word

matrix = convert_to_P(matrix, P_dist='conditional')

matrix.rename(index = {v:k for k,v in w2i.items() if k!='<unk>'}, inplace=True)

if args.verbose:
    eprint('Matrix ready')
    eprint('Writing to CSV')
    
if args.outfile == None:
    filename_w_ext = os.path.basename(conllfile)
    filename, _ = os.path.splitext(filename_w_ext)
    filename += '_depdist.csv'
else:
    filename = args.outfile
matrix.to_csv(filename, index=True)

if args.verbose:
    eprint('Matrix saved as {}'.format(filename))
    eprint('Done')

if args.timer:
    end = time.time()
    elapsed = round(end - start,4)
    eprint('Time elapsed: {} seconds'.format(elapsed))
