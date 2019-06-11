import numpy as np
from numpy.linalg import norm
import copy
from itertools import islice
import argparse
from sys import stdout,stderr
from time import ctime, time

argparser = argparse.ArgumentParser('For making a dictionary between languages')
argparser.add_argument('src_embs', help='the source language embeddings')
argparser.add_argument('tgt_embs', help='the target language embeddings')
argparser.add_argument('outfile', help='the file that the dictionary should be printed into')
argparser.add_argument('-norm_sequence', help='sequence of normalisation functions to perform on the embeddings.', nargs='*',
                       type=str, default=('length_normalise','mean_center','length_normalise'))
argparser.add_argument('-topN', action='store', type=int, help='top N items from the dictionary to keep', default=1000)
argparser.add_argument('-threshold', '-thr', action='store', type=float, help='proximity threshold to keep entries', default=0.5)
argparser.add_argument('--symmetry', '-sym', action='store', type=int, default=1, help='symmetry constraint; default True')
argparser.add_argument('--verbose', '-v', action='store', type=int, default=0, help='verbosity (default False')
args = argparser.parse_args()



def read_embeddings(filepath):
    with open(filepath) as f:
        #header in w2v format, which encodes vocabulary size and number of dimensions
        try:
            nrow, dims = f.readline().strip().split()
            nrow, dims = int(nrow),int(dims)
        except ValueError:
            print('Header missing.  Please put in word2vec format header with this format: m_rows n_dims')
            quit()
        vocab = []
        matrix = np.empty((nrow, dims), dtype='float')
        i = 0
        while i <= nrow:
            array = f.readline().strip()
            if len(array)==0: #exceeded number of rows or something has gone wrong
                break
            array = array.split()
            dim_ind = len(array)-dims
            word, vec = array[:dim_ind], array[dim_ind:]
            word = ' '.join(word)
            vocab.append(word)
            matrix[i] = vec
            i+=1
    return vocab,matrix
            
"""src_words = ['the','cat','sleeps','drinks']
src_vecs = np.array([[0.1,-1.5,0.8],
                     [0.4,-0.1,-1.5],
                     [0.6,0.4,0.5],
                     [0.55,0.5,0.01]])

tgt_words = ['le','chat','chien','dort']
tgt_vecs = np.array([[0.15,-1.1,0.99],
                     [0.2,-0.19,-1.2],
                     [0.15,-0.15,-0.98],
                     [0.7,0.35,0.6]])"""

def matrix_normalise(matrix):
    """performs matrix normalisation on the embedding space"""
    return matrix / np.linalg.norm(matrix)

def length_normalise(matrix):
    """vector length normalisation"""
    norms = np.linalg.norm(matrix, axis=1)[:,None]
    norms[norms==0] = 1
    return matrix / norms

def mean_center(matrix):
    """mean centering of embedding space"""
    mu = np.mean(matrix, axis=0)
    return matrix-mu

def dim_length_normalise(matrix):
    """dimension length normalisation across vector space"""
    norms = np.linalg.norm(matrix, axis=0)
    norms[norms==0] = 1
    return matrix / norms

def dim_mean_center(matrix):
    """mean centering of embedding space"""
    mu = np.mean(matrix, axis=1)[:,None]
    return matrix-mu

def norm_sequence(matrix, pipeline):
    """pipeline for normalising word embeddings
    
    Args:
    matrix: a dependency matrix
    *pipeline: a sequence of one or more normalisation functions as defined above
    example:
    >>>norm_sequence(matrix, (length_normalise, mean_center, length_normalise))
    [performs length norm, mean center, length norm on matrix and returns]"""

    funcs = {'matrix_normalise': matrix_normalise,
             'length_normalise': length_normalise,
             'mean_center': mean_center,
             'dim_length_normalise': dim_length_normalise,
             'dim_mean_center': dim_mean_center}
    
    for op in pipeline:
        fn = funcs[op]
        matrix = fn(matrix)
    return matrix





#step 1 & 2: Compute distances for each word in source language and target language
def compute_distances(src_words, src_vecs, tgt_words, tgt_vecs, verbose=0):
    src_neighbours = {}
    tgt_neighbours = {}
    sl, sm = int(len(src_words)), int(len(src_words)/20)
    tl, tm = int(len(src_words)), int(len(tgt_words)/20)
    i=0
    for word,vec in zip(src_words, src_vecs):
        distances = vec @ tgt_vecs.T #dot product word and all tgt words. T=transpose
        closest = np.argmax(distances) #gets an index of the closest tgt vec
        src_neighbours[word] = tgt_words[closest], distances[closest]
        if verbose and sm>0:
            if i % sm == 0:
                print('distances found for {} out of {} source language words'.format(i,sl))
        i+=1
            
    i=0
    for word,vec in zip(tgt_words, tgt_vecs):
        distances = vec @ src_vecs.T #dot product word and all tgt words. T=transpose
        closest = np.argmax(distances) #gets an index of the closest tgt vec
        tgt_neighbours[word] = src_words[closest], distances[closest]
        if verbose and tm>0:
            if i % tm == 0:
                print('distances found for {} out of {} target language words'.format(i,tl))
        i+=1


    return src_neighbours, tgt_neighbours


#step 3: Find bilingual neighbours
#step 3.1 (optional: symmetry constraint)
#step 3.2 (optional: closeness threshold)
def find_neighbours(src_words, src_vecs, tgt_words, tgt_vecs, src_neighbours, tgt_neighbours, SYM=True, THR=0.5):
    bidict = [] #The bilingual dictionary
    q_src = src_words.copy()
    q_tgt = tgt_words.copy()
    if SYM: #If symmetry constraint is True, we can save a lot of time because we only have to look at one word list
        for w_src in q_src:
            w_tgt, s2t_dist = src_neighbours[w_src]
            tgt2src, t2s_dist = tgt_neighbours[w_tgt]
            if tgt2src == w_src:
                bidict.append((w_src,w_tgt))
    else:
        for w_src in q_src:
            w_tgt, s2t_dist = src_neighbours[w_src]
            tgt2src, _ = tgt_neighbours[w_tgt]
            if s2t_dist > THR:
                bidict.append((w_src,w_tgt))
                if tgt2src == w_src:
                    q_tgt.remove(w_tgt) #Removes this word from the target words
        for w_tgt in q_tgt:
            w_src, t2s_dist = tgt_neighbours[w_tgt]
            if s2t_dist > THR:
                bidict.append((w_src,w_tgt))

    return bidict
        
#our bilingual dictionary
if __name__=='__main__':
    if args.verbose:
        print('Began process {}'.format(ctime()), file=stderr)
    src_words, src_vecs = read_embeddings(args.src_embs) #Read embs
    tgt_words, tgt_vecs = read_embeddings(args.tgt_embs)
    src_embs = norm_sequence(src_vecs, args.norm_sequence) #Normalise embbs
    tgt_embs = norm_sequence(tgt_vecs, args.norm_sequence)
    src_neighbours, tgt_neighbours = compute_distances(src_words, src_vecs, tgt_words, tgt_vecs, args.verbose) #compute distances and get neighbour dictionaries
    bidict = find_neighbours(src_words, src_vecs, tgt_words, tgt_vecs, src_neighbours, tgt_neighbours,
                             SYM=args.symmetry, THR=args.threshold) #make the dictionary
    outfile = open(args.outfile,'w') #print the dictionary
    for s,t in bidict:
        print(s,'\t',t, file=outfile)
