import numpy as np
import numpy.linalg as linalg
from scipy.stats import entropy
import pandas as pd
import time
from collections import defaultdict
from sys import stderr, stdout
import argparse

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 50)

argparser = argparse.ArgumentParser('For making a dictionary between languages')
argparser.add_argument('srcfile', help='the source language\'s matrix')
argparser.add_argument('tgtfile', help='the target language\'s matrix')
argparser.add_argument('outfile', help='the file that the dictionary should be printed into')
argparser.add_argument('-diverge', help='the measure of divergence. options: jsd, kl, cos', default='jsd')
argparser.add_argument('-batchsize', action='store', type=int, help='batch size in which the CSVs will be read.  higher is faster, but requires more memory', default=64)
argparser.add_argument('-topN', action='store', type=int, help='top N items from the dictionary to keep', default=100)
argparser.add_argument('--tgt2src', action='store_true', help='map from target language words to source language words. recommended if the target language is much smaller than the source language, but will produce a much smaller dictionary', default=100)
argparser.add_argument('--verbose', action='store_true', help='verbose setting')
args = argparser.parse_args()


def ent(P,Q=None,axis=0):
	    if Q is None:
		    return -np.sum(P * np.log(P), axis=axis)
	    else:
		    return np.sum(P*np.log(P/Q), axis=axis)

def normalise(arr, ord =0, axis=0):
    if len(arr.shape) == 1:
        return arr / linalg.norm(arr, ord=ord)
    elif len(arr.shape)==2:
        if axis==0:
            return arr / linalg.norm(arr, ord=ord, axis=0)
        elif axis==1:
            return arr / linalg.norm(arr, ord=ord, axis=1)[:,None] #broadcast to fit

def JSD(P, Q, ord=0, axis=0):
    axis_check = lambda x: 0 if len(x.shape) < 2 else axis
    _P = normalise(P, ord=ord, axis=axis_check(P))
    _Q = normalise(Q, ord=ord, axis=axis_check(Q))
    #print(_P)
    #print()
    #print(_Q)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (ent(_P, _M, axis=axis) + ent(_Q, _M, axis=axis))

src = args.srcfile
tgt = args.tgtfile

BATCHSIZE = args.batchsize
topN = args.topN
tgt2src = args.tgt2src
verbose = args.verbose

if tgt2src: #swap src and tgt, so that tgt is mapped to src
            temp = src
            src = tgt
            tgt = temp
            del temp
            

top_alignments = defaultdict(lambda: ('_',float('inf')))

src_reader = pd.read_csv(src, index_col=0, usecols=None, chunksize=BATCHSIZE)

print('start', time.ctime(), file=stderr)
counter = 0
for src_batch in src_reader:
            for w_src,vec_src in zip(src_batch.index, src_batch.to_numpy()):
                        tgt_reader = pd.read_csv(tgt, index_col=0, usecols=None, chunksize=BATCHSIZE)
                        for tgt_batch in tgt_reader:
                                    wrds_tgt, vecs_tgt = tgt_batch.index, tgt_batch.to_numpy()
                                    if args.diverge == 'jsd':
                                                diverge = JSD(vec_src, vecs_tgt, axis=1)
                                    elif args.diverge == 'kl':
                                                diverge = ent(vec_src, vecs_tgt, axis=1)
                                    elif args.diverge == 'cos':
                                                diverge = np.dot(vec_src, vecs_tgt.T)
                                    top_ind = np.argmin(diverge)
                                    lowest = np.min(diverge)
                                    if lowest < top_alignments[w_src][1]:
                                                top_alignments[w_src] = (wrds_tgt[top_ind], lowest)
            if verbose:
                        counter += 1
                        print('{} batches processed'.format(counter),file=stderr)
src_reader.close()
tgt_reader.close()
print('done', time.ctime(),file=stderr)

top_alignments = {k:v for k,v in top_alignments.items() if v[1] > 0.0}
if tgt2src:
            top_alignments = {v[0]:(k,v[1]) for k,v in top_alignments.items()}


with open(args.outfile,'w') as outfile:
	for k in sorted(top_alignments, key=lambda x: top_alignments[x][1])[:topN]:
        	print(k+'\t'+top_alignments[k][0], file=outfile)
