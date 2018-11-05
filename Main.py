

import os
from os.path import join as path_join
import datetime


import numpy as np
from numpy import dot, trace, array
from numpy.linalg import svd, norm, inv, eig
import gensim
from gensim.models.keyedvectors import KeyedVectors
import scipy.io as sio
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
import pandas as pd


FOLDER_DATA = path_join(os.getcwd(), "data")
FOLDER_DISTRIBUTIONALVEC = path_join(FOLDER_DATA, "DistributionalVec")
FNAME_WORD2VEC_GOOGLENEWS = path_join(FOLDER_DISTRIBUTIONALVEC, "GoogleNews-vectors-negative300.bin.gz")
FNAME_GLOVE_840B300D = path_join(FOLDER_DISTRIBUTIONALVEC, "gensim_glove.840B.300d.txt.bin")

FOLDER_NONDISTRIBUTIONALVEC = path_join(FOLDER_DATA, "NondistributionalVec")

FNAME_WIKIWORDS = path_join(FOLDER_DATA, "Wiki_vocab_gt200", "enwiki_vocab_min200.txt")

FOLDER_EVALUATION = path_join(FOLDER_DATA, "Evaluate")
FOLDER_EVALUATION_WORDVECTORSORG = path_join(FOLDER_EVALUATION, "wordvectors.org_word-sim")
LIST_FNAME_EVALUATION_WORDVECTORSORG = ['EN-RG-65.txt', 'EN-WS-353-ALL.txt', 'EN-RW-STANFORD.txt',
                                        'EN-MEN-TR-3k.txt', 'EN-MTurk-287.txt', 'EN-SIMLEX-999.txt',
                                        'EN-SimVerb-3500.txt']



def read_dist_wordvec(D_embedding_fname, fname_wordlist):
    '''
    To read embedding vectors and only keep those in wordlist.

    :param D_embedding_fname: {"embedding_name": "filepath/filename"}, where the value is the absolute filename of the
                              pretrained embedding. Currently only support those could be passed to
                              gensim.models.keyedvectors.KeyedVectors.load_word2vec_format with binary=True.
    :param fname_wordlist: A space separated txt whose first column is the word in wordlist and should be kept.

    :return: D_embedding_vec: {"embedding_name": (["word1", "word2", ...], numpy.array(...))}, whether the values are
                              tuples of size 2. The first element of the tuple is a list of words, and the second is
                              a 2-D numpy.array whose rows are corresponding to the vector representations of words in
                               wordlist, i.e. its.shape = (n_words, dim_embedding).
    '''

    wordlist = set()
    with open(fname_wordlist) as fread_wordlist:
        for line in fread_wordlist:
            wordlist.add(line.strip().split(" ")[0])

    D_embedding_vec = dict()
    # JS: What does ii mean -> i.e. name your variable so that I can understand
    # p.s. I know what it means, but in general, I would expect ii to be an
    # integer instead of a file name
    for ii in D_embedding_fname:
        wordVecModel = KeyedVectors.load_word2vec_format(D_embedding_fname[ii], binary=True)
        word_in_wordlist_and_model = set(list(wordVecModel.vocab)).intersection(wordlist)
        x_collector_indices = []
        x_collector_words = []

        for word in word_in_wordlist_and_model:
            x_collector_indices.append(wordVecModel.vocab[word].index)
            x_collector_words.append(word)

        x_collector = wordVecModel.vectors[x_collector_indices, :]
        #print(np.mean(x_collector,0))
        #x_collector -= np.mean(x_collector,0)

        D_embedding_vec[ii] = (x_collector_words, x_collector)

    return D_embedding_vec


def read_nondist_wordvec():
    pass


def proto_conceptor(word_vec, embedding_name, alpha = 1, plotSpectrum = False):
    # compute the prototype conceptor with alpha = 1

    x_collector = word_vec.T

    nrWords = x_collector.shape[1]  # number of total words

    R = x_collector.dot(x_collector.T) / nrWords  # calculate the correlation matrix

    C = R @ inv(R + alpha ** (-2) * np.eye(300))  # calculate the conceptor matrix

    if plotSpectrum:  # visualization: plot the spectrum of the correlation matrix
        Ux, Sx, _ = np.linalg.svd(R)

        downWeighedSigVal = Sx / np.array([(1 + alpha * sigma2) for sigma2 in Sx])

        plt.plot(np.arange(300), Sx, 'bo', alpha=0.4,
                 label='orig ' + embedding_name + ' spectrum')  # here alpha is the transparency level for dots, don't get confused by the hyperparameter alpha!
        plt.plot(np.arange(300), downWeighedSigVal, 'ro', alpha=0.4,
                 label='downweighted ' + embedding_name + ' spectrum')  # here alpha is the transparency level for dots, don't get confused by the hyperparameter alpha!

        plt.legend()
        plt.show()
    return C


def PHI(C, gamma):
    # The PHI function allows us to tune the aperture (alpha) by reusing the prototype conceptor. See https://arxiv.org/pdf/1403.3369.pdf
    dim = C.shape[0]
    if gamma == 0:
        U, S, _ = np.linalg.svd(C)
        S[S < 1] = np.zeros((np.sum((S < 1).astype(float)), 1))
        C_new = U.dot(S).dot(U.T)
    elif gamma == np.Inf:
        U, S, _ = np.linalg.svd(C)
        S[S > 0] = np.zeros((np.sum((S > 0).astype(float)), 1))
        C_new = U.dot(S).dot(U.T)
    else:
        C_new = C.dot(np.linalg.inv(C + gamma ** -2 * (np.eye(dim) - C)))

    return C_new


def load_abtt_results():
    ## (Optional) Load the results reported in https://openreview.net/pdf?id=HkuGJ3kCb
    w2v_abtt_result = {}
    w2v_abtt_result['EN-RG-65.txt'] = 0.7834
    w2v_abtt_result['EN-WS-353-ALL.txt'] = 0.6905
    w2v_abtt_result['EN-RW-STANFORD.txt'] = 0.5433
    w2v_abtt_result['EN-MEN-TR-3k.txt'] = 0.7908
    w2v_abtt_result['EN-MTurk-287.txt'] = 0.6935
    w2v_abtt_result['EN-SIMLEX-999.txt'] = 0.4501
    w2v_abtt_result['EN-SimVerb-3500.txt'] = 0.3650

    glove_abtt_result = {}
    glove_abtt_result['EN-RG-65.txt'] = 0.7436
    glove_abtt_result['EN-WS-353-ALL.txt'] = 0.7679
    glove_abtt_result['EN-RW-STANFORD.txt'] = 0.5204
    glove_abtt_result['EN-MEN-TR-3k.txt'] = 0.8178
    glove_abtt_result['EN-MTurk-287.txt'] = 0.7085
    glove_abtt_result['EN-SIMLEX-999.txt'] = 0.4497
    glove_abtt_result['EN-SimVerb-3500.txt'] = 0.3223

    return w2v_abtt_result, glove_abtt_result


def similarity_eval(dataSetAddress, wordVecModel, vocab, conceptorProj=False, C = None):

    D_vocab_index = {i: c for c, i in enumerate(vocab)}

    pair_list = []
    with open(dataSetAddress, "r") as fread_simlex:
        for line in fread_simlex:
            tokens = line.split()
            word_i = tokens[0]
            word_j = tokens[1]
            score = float(tokens[2])
            if word_i in D_vocab_index and word_j in D_vocab_index:
                pair_list.append(((word_i, word_j), score))

    pair_list.sort(key=lambda x: - x[1])  # order the pairs from highest score (most similar) to lowest score (least similar)

    extracted_scores = {}

    extracted_list = []

    C = C if conceptorProj else np.zeros((300, 300))

    for (x, y) in pair_list:
        (word_i, word_j) = x

        current_distance = cosine(wordVecModel[D_vocab_index[word_i]] - np.matmul(C, wordVecModel[D_vocab_index[word_i]]),
                                  wordVecModel[D_vocab_index[word_j]] - np.matmul(C, wordVecModel[D_vocab_index[word_j]]))
        extracted_scores[(word_i, word_j)] = current_distance
        extracted_list.append(((word_i, word_j), current_distance))

    extracted_list.sort(key=lambda x: x[1])

    spearman_original_list = []
    spearman_target_list = []

    for position_1, (word_pair, score_1) in enumerate(pair_list):
        score_2 = extracted_scores[word_pair]
        position_2 = extracted_list.index((word_pair, score_2))
        spearman_original_list.append(position_1)
        spearman_target_list.append(position_2)

    spearman_rho = spearmanr(spearman_original_list, spearman_target_list)

    return spearman_rho[0]


def evaluation_wordvectorsorg(D_embedding_vec, folder_evaluation, L_fname):

    D_embedding_abbt_result = dict()
    D_embedding_abbt_result["word2vec"], D_embedding_abbt_result["glove840B300D"] = load_abtt_results()

    D_embedding_beta = {"word2vec": 2, "glove840B300D": 1}
    D_embedding_Cproto = dict()
    D_embedding_Cadjusted = dict()
    # JS: Again ... not a fan of ii
    for ii in D_embedding_vec:
        D_embedding_Cproto[ii] = proto_conceptor(D_embedding_vec[ii][1], ii)
        D_embedding_Cadjusted[ii] = PHI(D_embedding_Cproto[ii], D_embedding_beta[ii])
        print('beta of % s = % s' % (ii, D_embedding_beta[ii]))
        print('Quota for % s conceptor is' % ii, trace(D_embedding_Cadjusted[ii]) / 300)
    print()

    wordSimResult = {}
    for fname in L_fname:
        dataSetAddress = path_join(folder_evaluation, fname)
        print('evaluating the data set', fname)

        for embedding in D_embedding_vec:

            print('% s + ABTT %.4f: ' % (embedding, D_embedding_abbt_result[embedding][fname]))

            tmp_similarity = similarity_eval(dataSetAddress,
                                             wordVecModel = D_embedding_vec[embedding][1],
                                             vocab = D_embedding_vec[embedding][0],
                                             conceptorProj = True,
                                             C = D_embedding_Cadjusted[embedding])

            print('% s + conceptor : %.4f' % (embedding, tmp_similarity))
            wordSimResult["% s- % s" % (embedding, fname.split(".")[0])] = \
                [D_embedding_abbt_result[embedding][fname] * 100,
                 round(tmp_similarity * 100, 2)]


        print('\n')

    wordSimResult_df = pd.DataFrame(wordSimResult, index=['all-but-the-top', 'conceptor']).T
    ax = wordSimResult_df.plot(kind="bar")
    ax.legend(loc="best")
    ax.set_ylim(20, 100)
    ax.set_ylabel("Pearson correlation coefficient x 100")
    plt.show()

    return


def Main():

    print("start running conceptor on nondistributional word vectors")
    now_time = datetime.datetime.now
    start_time = now_time()
    print()

    print("start reading distributional embeddings", (now_time() - start_time).seconds)
    D_embedding_vec = read_dist_wordvec({"word2vec": FNAME_WORD2VEC_GOOGLENEWS,
                                         "glove840B300D": FNAME_GLOVE_840B300D},
                                        FNAME_WIKIWORDS)
    print("finished reading distributional embeddings", (now_time() - start_time).seconds)

    #for ii in D_embedding_vec: proto_conceptor(D_embedding_vec[ii][1], ii, plotSpectrum = True)

    evaluation_wordvectorsorg(D_embedding_vec, FOLDER_EVALUATION_WORDVECTORSORG, LIST_FNAME_EVALUATION_WORDVECTORSORG)

    print()
    print("finished running conceptor on nondistributional word vectors")
    print("total running time =", (now_time() - start_time).seconds)
    return


if __name__ == "__main__":
    Main()
