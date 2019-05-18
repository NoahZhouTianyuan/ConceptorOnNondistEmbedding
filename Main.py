

import os
from os.path import join as path_join, exists
import datetime
import gzip
from itertools import chain
import csv


import numpy as np
from numpy import dot, trace, array
from numpy.linalg import svd, norm, inv, eig
from numpy.random import randint, normal
import gensim
from gensim.models.keyedvectors import KeyedVectors
import scipy.io as sio
from scipy.stats import spearmanr, rankdata
from scipy.spatial.distance import cosine
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
import pandas as pd


FOLDER_RES = path_join(os.getcwd(), "res")

FOLDER_DATA = path_join(os.getcwd(), "data")
FOLDER_DISTRIBUTIONALVEC = path_join(FOLDER_DATA, "DistributionalVec")
FOLDER_LEXICONS = path_join(FOLDER_DATA, "lexicons")
FNAME_WORD2VEC_GOOGLENEWS = path_join(FOLDER_DISTRIBUTIONALVEC, "GoogleNews-vectors-negative300.bin.gz")
FNAME_GLOVE_840B300D = path_join(FOLDER_DISTRIBUTIONALVEC, "gensim_glove.840B.300d.txt.bin")
FNAME_FASTTEXT_CRAWL = path_join(FOLDER_DISTRIBUTIONALVEC, "crawl-300d-2M.vec")
FNAME_FASTTEXT_WIKINEWS = path_join(FOLDER_DISTRIBUTIONALVEC, "wiki-news-300d-1M.vec")

#FOLDER_NONDISTRIBUTIONALVEC = r"D:\Work\JRodu\WordEmbeddingDomainSpecific\data source\pretrained embedding\nondist\res"
FOLDER_NONDISTRIBUTIONALVEC = path_join(FOLDER_DATA, "NondistributionalVec")

FOLDER_MYVEC = path_join(FOLDER_DATA, "MyVec")

FNAME_WIKIWORDS = path_join(FOLDER_DATA, "Wiki_vocab_gt200", "enwiki_vocab_min200.txt")

FOLDER_EVALUATION = path_join(FOLDER_DATA, "Evaluate")
FOLDER_EVALUATION_WORDVECTORSORG = path_join(FOLDER_EVALUATION, "wordvectors.org_word-sim")
LIST_FNAME_EVALUATION_WORDVECTORSORG = ['EN-RG-65.txt', 'EN-WS-353-ALL.txt',
                                        'EN-RW-STANFORD.txt',
                                        'EN-MEN-TR-3k.txt', 'EN-MTurk-287.txt', 'EN-SIMLEX-999.txt',
                                        'EN-SimVerb-3500.txt']

FOLDER_EVALUATION_CATEGORIZATION = path_join(FOLDER_EVALUATION, "concept_categorization")
LIST_FNAME_EVALUATION_CATEGORIZATION = ['battig.csv', 'ap.csv', 'essli-2008.csv']

'''
DICT_NONDIST_EMBEDDING_FNAME = {"Nondist300D_fullSV": "Nondist300D_fullSV.txt",
                                "Nondist300D_halfSV": "Nondist300D_halfSV.txt",
                                "Nondist300D_noSV": "Nondist300D_noSV.txt",
                                "Nondist300D_wiki200_fullSV": "Nondist300D_fullSV.txt",
                                "Nondist300D_wiki200_halfSV": "Nondist300D_halfSV.txt",
                                "Nondist300D_wiki200_noSV": "Nondist300D_noSV.txt",}
DICT_NONDIST_EMBEDDING_FNAME = dict()
for sv_type in ["fullSV", "halfSV", "noSV"]:
    for wiki200_type in ["_wiki200", ""]:
        for dim in [100 * i for i in range(1, 11)]:
            tmp_embedding = "Nondist_% sD_% s% s" % (dim, sv_type, wiki200_type)
            DICT_NONDIST_EMBEDDING_FNAME[tmp_embedding] = tmp_embedding + ".txt"
'''

'''
DICT_NONDIST_EMBEDDING_FNAME = dict()
for sv_type in ["fullSV", "halfSV", "noSV"]:
    for wiki200_type in ["_wiki200", ""]:
        for dist_embedding in ["fasttextCrawl", "glove840B300D", "word2vec"]:
            tmp_embedding = "Nondist_300D_% s% s-to-% s+conceptored" % (sv_type, wiki200_type, dist_embedding)
            DICT_NONDIST_EMBEDDING_FNAME[tmp_embedding] = tmp_embedding + ".txt"
            tmp_embedding = "Nondist_300D_% s% s+conceptored-to-% s+conceptored" % (sv_type, wiki200_type, dist_embedding)
            DICT_NONDIST_EMBEDDING_FNAME[tmp_embedding] = tmp_embedding + ".txt"
            tmp_embedding = "Nondist_300D_% s% s-to-% s" % (sv_type, wiki200_type, dist_embedding)
            DICT_NONDIST_EMBEDDING_FNAME[tmp_embedding] = tmp_embedding + ".txt"
            tmp_embedding = "Nondist_300D_% s% s+conceptored-to-% s" % (sv_type, wiki200_type, dist_embedding)
            DICT_NONDIST_EMBEDDING_FNAME[tmp_embedding] = tmp_embedding + ".txt"
for sv_type in ["fullSV", "halfSV", "noSV"]:
    for wiki200_type in ["_wiki200", ""]:
        for dim in [100 * i for i in range(1, 11)]:
            tmp_embedding = "Nondist_% sD_% s% s" % (dim, sv_type, wiki200_type)
            DICT_NONDIST_EMBEDDING_FNAME[tmp_embedding] = tmp_embedding + ".txt"
'''

'''
DICT_NONDIST_EMBEDDING_FNAME = dict()
for sv_type in ["fullSV", "halfSV", "noSV"]:
    for wiki200_type in ["_wiki200freq"]:
        for dim in [300]:
            tmp_embedding = "Nondist_% sD_% s% s" % (dim, sv_type, wiki200_type)
            DICT_NONDIST_EMBEDDING_FNAME[tmp_embedding] = tmp_embedding + ".txt"
'''

FOLDER_ALIGNED_DISTRIBUTIONALVEC = r"D:\Work\JRodu\ConceptorOnNondist\MUSE\MUSE-master\res_myAlign"
FOLDER_ALIGNED_NONDISTRIBUTIONALVEC = r"D:\Work\JRodu\ConceptorOnNondist\MUSE\MUSE-master\res_myAlign"
DICT_ALIGNED_DIST_EMBEDDING_FNAME = dict()
DICT_ALIGNED_NONDIST_EMBEDDING_FNAME = dict()
L_nondist_conceptored = ["Nondist_300D_fullSV+conceptored.txt",
                         "Nondist_300D_halfSV+conceptored.txt",
                         "Nondist_300D_noSV+conceptored.txt",
                         "Nondist_300D_fullSV_wiki200+conceptored.txt",
                         "Nondist_300D_halfSV_wiki200+conceptored.txt",
                         "Nondist_300D_noSV_wiki200+conceptored.txt",
                         "Nondist_300D_fullSV_wiki200freq+conceptored.txt",
                         "Nondist_300D_halfSV_wiki200freq+conceptored.txt",
                         "Nondist_300D_noSV_wiki200freq+conceptored.txt"]
L_nondist = ["Nondist_300D_fullSV.txt",
             "Nondist_300D_halfSV.txt",
             "Nondist_300D_noSV.txt",
             "Nondist_300D_fullSV_wiki200.txt",
             "Nondist_300D_halfSV_wiki200.txt",
             "Nondist_300D_noSV_wiki200.txt",
             "Nondist_300D_fullSV_wiki200freq.txt",
             "Nondist_300D_halfSV_wiki200freq.txt",
             "Nondist_300D_noSV_wiki200freq.txt"]
L_dist = ["fasttextCrawl.txt",
          "glove840B300D.txt",
          "word2vec.txt"]
L_dist_conceptored = ["fasttextCrawl+conceptored.txt",
          "glove840B300D+conceptored.txt",
          "word2vec+conceptored.txt"]
for embed_nondist in chain(L_nondist, L_nondist_conceptored):
    for embed_dist in chain(L_dist, L_dist_conceptored):
        tmp_embed = "% s-by-% s.txt" % (embed_dist.split(".txt")[0], embed_nondist.split(".txt")[0])
        DICT_ALIGNED_DIST_EMBEDDING_FNAME[tmp_embed] = tmp_embed
        tmp_embed = "% s-by-% s.txt" % (embed_nondist.split(".txt")[0], embed_dist.split(".txt")[0])
        DICT_ALIGNED_NONDIST_EMBEDDING_FNAME[tmp_embed] = tmp_embed


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
    with open(fname_wordlist) as f_wordlist:
        for line in f_wordlist:
            wordlist.add(line.strip().split(" ")[0])

    D_embedding_vec = dict()
    for embedding in D_embedding_fname:

        if ".bin" in D_embedding_fname[embedding]:
            wordVecModel = KeyedVectors.load_word2vec_format(D_embedding_fname[embedding], binary=True)
            word_in_wordlist_and_model = set(list(wordVecModel.vocab)).intersection(wordlist)
            x_collector_indices = []
            x_collector_words = []

            for word in word_in_wordlist_and_model:
                x_collector_indices.append(wordVecModel.vocab[word].index)
                x_collector_words.append(word)

            x_collector = wordVecModel.vectors[x_collector_indices, :]
            #print(np.mean(x_collector,0))
            #x_collector -= np.mean(x_collector,0)
        elif ".txt" == D_embedding_fname[embedding][-4:] or ".vec" == D_embedding_fname[embedding][-4:]:
            x_collector_words, x_collector = [], []
            with open(D_embedding_fname[embedding], encoding = "gbk", errors = "ignore") as f_embedding:
                for cc, line in enumerate(f_embedding):
                    if cc == 0:
                        tmp = line.strip().split(" ")
                        if len(tmp) == 2 and tmp[0].isdigit() and tmp[1].isdigit():
                            continue
                    line = line.strip().split(" ")
                    if line[0] not in wordlist:
                        continue
                    x_collector_words.append(line[0])
                    x_collector.append(array(line[1: ], dtype = np.float32))
            x_collector = np.array(x_collector)

        else:
            print("Error: Unknown type of embedding file % s." % D_embedding_fname[embedding])
            continue

        D_embedding_vec[embedding] = (x_collector_words, x_collector)

    return D_embedding_vec


def read_nondist_wordvec(folder_data, fname_wordlist, D_embedding_fname, is_read_dense = True):
    '''
    To read nondistributional word vectors. Offer two option - dense or sparse.
    :param folder_data: The data folder storing existed word vectors. It SHOULD CONTAIN two files, one is
                        "Nondist300D.txt" for dense, and the other is "binary-vectors.txt.gz" for sparse.
    :param fname_wordlist: A space separated txt whose first column is the word in wordlist and should be kept.
    :param is_read_dense: if True, read dense vectors, otherwise read sparse ones.
    :return: vocab, vector_matrix. vocab is a list of words, and vector_matrix is a matrix whose rows are vector
             representations of corresponding words in vocab. If dense, vector_matrix is of np.array, otherwise it
             is of scipy.sparse.csc_matrix.
    '''
    wordlist = set()
    with open(fname_wordlist) as f_wordlist:
        for line in f_wordlist:
            wordlist.add(line.strip().split(" ")[0])

    if is_read_dense:
        ans = dict()
        for embedding in D_embedding_fname:
            vocab, X = [], []
            if not exists(path_join(folder_data, D_embedding_fname[embedding])):
                print("Error: no embedding file of % s: % s. Skipped." % (embedding, D_embedding_fname[embedding]))
            with open(path_join(folder_data, D_embedding_fname[embedding]), encoding = "gbk", errors="ignore") as f_embedding:
                for cc, line in enumerate(f_embedding):
                    if cc == 0:
                        tmp = line.strip().split(" ")
                        if len(tmp) == 2 and tmp[0].isdigit() and tmp[1].isdigit():
                            continue
                    line = line.strip().split(" ")
                    if line[0] not in wordlist:
                        continue
                    vocab.append(line[0])
                    X.append(array(line[1: ], dtype = np.float32))
            X = array(X)
            ans[embedding] = (vocab, X)

    else:

        '''
        ans = dict()
        vocab, data, row_ind, col_ind = [], [], [], []
        with gzip.open(path_join(folder_data, "binary-vectors.txt.gz")) as f_sparseEmbedding:
            for cc, line in enumerate(f_sparseEmbedding):
                line = line.decode("gbk").strip().split(" ")

                if line[0] not in wordlist:
                    continue
                vocab.append(line[0])
                for col, ii in enumerate(line[1:]):
                    if float(ii) == 1:
                        row_ind.append(len(vocab) - 1)
                        col_ind.append(col)
                        data.append(1)
            ncol = len(line) - 1

        X = csr_matrix((data, (row_ind, col_ind)),
                       shape=(len(vocab), ncol),
                       dtype=float)
        ans["sparse_embedding"] = (vocab, X)
        '''

        folder_lex = FOLDER_LEXICONS
        ans = dict()

        D_fname_feat = dict()
        D_feat_fname = dict()
        vectors = {}
        all_feat = {}
        L_fname_lex = os.listdir(folder_lex)
        for fname in L_fname_lex:
            # print(fname)
            D_fname_feat[fname.split(".txt")[0]] = set()
            with open(path_join(folder_lex, fname)) as f_lex:
                for line in f_lex:
                    things = line.strip().lower().split()
                    pos_present = False
                    try:
                        word, pos = things[0].split('.')
                        pos_present = True
                    except:
                        word = things[0]
                    if word not in vectors:
                        vectors[word] = {}
                    for feat in things[1:]:
                        if pos_present:
                            feat = feat + '.' + pos
                        vectors[word][feat] = 1
                        if feat not in all_feat:
                            all_feat[feat] = len(all_feat)
                        D_fname_feat[fname.split(".txt")[0]].add(feat)
                        D_feat_fname[feat] = fname.split(".txt")[0]

        L_word, data, row_ind, col_ind = [], [], [], []
        for cc, word in enumerate(vectors):
            L_word.append(word)
            for feat in vectors[word]:
                data.append(1)
                col_ind.append(all_feat[feat])
                row_ind.append(len(L_word) - 1)
            # if cc % 1000 == 0:
            #    print(cc, (now_time() - start_time).seconds)
        # print(cc, (now_time() - start_time).seconds)

        ncol = len(all_feat)
        nrow = len(L_word)
        A = csr_matrix((data, (row_ind, col_ind)),
                       shape=(len(L_word), ncol),
                       dtype=float)

        ans["sparse_embedding"] = (L_word, A)

    return ans


def proto_conceptor(word_vec, embedding_name, alpha = 1, plotSpectrum = False):
    # compute the prototype conceptor with alpha = 1

    x_collector = word_vec.T

    nrWords = x_collector.shape[1]  # number of total words
    dim = x_collector.shape[0]

    R = x_collector.dot(x_collector.T) / nrWords  # calculate the correlation matrix

    C = R @ inv(R + alpha ** (-2) * np.eye(dim))  # calculate the conceptor matrix

    if plotSpectrum:  # visualization: plot the spectrum of the correlation matrix
        Ux, Sx, _ = np.linalg.svd(R)

        downWeighedSigVal = Sx / np.array([(1 + alpha * sigma2) for sigma2 in Sx])

        plt.plot(np.arange(dim), Sx, 'bo', alpha=0.4,
                 label='orig ' + embedding_name + ' spectrum')  # here alpha is the transparency level for dots, don't get confused by the hyperparameter alpha!
        plt.plot(np.arange(dim), downWeighedSigVal, 'ro', alpha=0.4,
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


def abtt(word_vec, D = 1, is_centering = True):
    # D: the number of PCs to delete.
    tmp_mean = word_vec.mean(axis = 0)
    if is_centering:
        word_vec -= tmp_mean
    #u, s, vt = svds(word_vec, k = word_vec.shape[1] - 1)
    # k should be word_vec.shape[1], but svds doesn't support full-rank, i.e. k < min(word_vec.shape).
    u, s, vt = svd(word_vec, full_matrices = False)
    for ii in range(D):
        s[ii] = 0
    ans = u @ np.diag(s) @ vt
    if is_centering:
        ans += tmp_mean
    return ans


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


def perturb_magnitude(X):
    def tmp_cos(x1, x2):
        return x1 @ x2 / (norm(x1) * norm(x2))
    res = []
    for cc, tmp_line in enumerate(X):
        res.append(tmp_line * (1 + normal(0, 0.1, size = 1)))
        #res.append(tmp_line * 1.1)
        #print(tmp_line[:5], res[-1][:5], res[-1][:5] / tmp_line[:5], len(tmp_line))
    with open("tmp_log.txt", "a") as fw:
        fw.write(" ".join(map(str, [cosine_similarity([res[1]], [res[2]]), cosine_similarity([res[4]], [res[5]]), cosine_similarity([res[100]], [res[200]]),
                                    cosine_similarity([res[6]], [res[3]]), cosine_similarity([res[1]], [res[1000]])])) + "\n")
        fw.write(" ".join(map(str, [cosine(res[1], res[2]), cosine(res[4], res[5]), cosine(res[100], res[200]),
                                    cosine(res[6], res[3]), cosine(res[1], res[1000])])) + "\n")
        fw.write(" ".join(map(str, [tmp_cos(res[1], res[2]), tmp_cos(res[4], res[5]), tmp_cos(res[100], res[200]),
                                    tmp_cos(res[6], res[3]), tmp_cos(res[1], res[1000])])) + "\n")

    return array(res)


def similarity_eval(dataSetAddress, wordVecModel, vocab, conceptorProj=False, C = None):

    D_vocab_index = {i: c for c, i in enumerate(vocab)}

    pair_list = []
    with open(dataSetAddress, "r") as fread_simlex:
        for cc, line in enumerate(fread_simlex):
            #if cc == 0: continue
            tokens = line.split()
            word_i = tokens[0]
            word_j = tokens[1]
            score = float(tokens[2])
            if word_i in D_vocab_index and word_j in D_vocab_index:
                pair_list.append(((word_i, word_j), score))

    pair_list.sort(key=lambda x: - x[1])  # order the pairs from highest score (most similar) to lowest score (least similar)

    extracted_scores = {}
    extracted_scores_test_perturbation = {}

    extracted_list = []
    extracted_list_test_perturbation = []

    C = C if conceptorProj else np.zeros((wordVecModel.shape[1], wordVecModel.shape[1]))

    for (x, y) in pair_list:
        (word_i, word_j) = x

        current_distance = cosine(wordVecModel[D_vocab_index[word_i]] - np.matmul(C, wordVecModel[D_vocab_index[word_i]]),
                                  wordVecModel[D_vocab_index[word_j]] - np.matmul(C, wordVecModel[D_vocab_index[word_j]]))
        extracted_scores[(word_i, word_j)] = current_distance

        current_distance_test_perturbation = cosine(
            (wordVecModel[D_vocab_index[word_i]] - np.matmul(C, wordVecModel[D_vocab_index[word_i]])) * (1 + normal(0, 0.1, 1)) / 10000,
            (wordVecModel[D_vocab_index[word_j]] - np.matmul(C, wordVecModel[D_vocab_index[word_j]])) * (1 + normal(0, 0.1, 1)) / 10000)
        extracted_scores_test_perturbation[(word_i, word_j)] = current_distance_test_perturbation

        extracted_list.append(((word_i, word_j), current_distance))
        extracted_list_test_perturbation.append(((word_i, word_j), current_distance_test_perturbation))
        if abs(current_distance_test_perturbation - current_distance) > 1e-8:
            print("large dev in cos:", abs(current_distance_test_perturbation - current_distance))

    extracted_list.sort(key=lambda x: x[1])
    extracted_list_test_perturbation.sort(key=lambda x: x[1])

    spearman_original_list = []
    spearman_target_list = []

    for position_1, (word_pair, score_1) in enumerate(pair_list):
        score_2 = extracted_scores[word_pair]
        position_2 = extracted_list.index((word_pair, score_2))
        spearman_original_list.append(position_1)
        spearman_target_list.append(position_2)

    spearman_rho = spearmanr(spearman_original_list, spearman_target_list)

    spearman_original_list_test_perturbation = []
    spearman_target_list_test_perturbation = []

    for position_1, (word_pair, score_1) in enumerate(pair_list):
        score_2 = extracted_scores_test_perturbation[word_pair]
        position_2 = extracted_list_test_perturbation.index((word_pair, score_2))
        spearman_original_list_test_perturbation.append(position_1)
        spearman_target_list_test_perturbation.append(position_2)

    spearman_rho_test_perturbation = spearmanr(spearman_original_list_test_perturbation, spearman_target_list_test_perturbation)

    if spearman_rho_test_perturbation[0] != spearman_rho[0]:
        print("different spearman_rho", spearman_rho_test_perturbation[0], spearman_rho[0])

    #tmp = list(zip(*extracted_list))[1]
    #L_diff = []
    #for ii in range(len(tmp) - 1):
    #    L_diff.append(tmp[ii + 1] - tmp[ii])
    #L_diff.sort()
    #print("minimal_difference:", L_diff[:6])

    return spearman_rho[0]


def my_spearmanr(original_list, target_list):
    original_list = rankdata(original_list)
    target_list = rankdata(target_list)
    return -spearmanr(original_list, target_list)[0]


def similarity_eval_new(dataSetAddress, wordVecModel, vocab, conceptorProj=False, C = None):

    D_vocab_index = {i: c for c, i in enumerate(vocab)}

    pair_list = []
    with open(dataSetAddress, "r", encoding='utf-8', ) as fread_simlex:
        for cc, line in enumerate(fread_simlex):
            #if cc == 0: continue
            tokens = line.split()
            word_i = tokens[0]
            word_j = tokens[1]
            score = float(tokens[2])
            if word_i in D_vocab_index and word_j in D_vocab_index:
                pair_list.append(((word_i, word_j), score))
            else:
                pass
                #print("no this pair", word_i, word_j, score, dataSetAddress)

    original_list, target_list = [], []

    C = C if conceptorProj else np.zeros((wordVecModel.shape[1], wordVecModel.shape[1]))

    for (word_i, word_j), score in pair_list:
        current_distance = cosine(wordVecModel[D_vocab_index[word_i]] - np.matmul(C, wordVecModel[D_vocab_index[word_i]]),
                                  wordVecModel[D_vocab_index[word_j]] - np.matmul(C, wordVecModel[D_vocab_index[word_j]]))
        original_list.append(score)
        target_list.append(current_distance)

    spearman_rho = my_spearmanr(original_list, target_list)

    return spearman_rho


def similarity_eval_tl(dataSetAddress, wordVecModel, vocab, conceptorProj=False, C = None):

    dataset = []
    with open(dataSetAddress, encoding='utf-8', ) as fin:
        for line in fin:
            tokens = line.rstrip().split()
            if tokens[0] in vocab and tokens[1] in vocab:
                dataset.append(((tokens[0], tokens[1]), float(tokens[2])))
    dataset.sort(key = lambda score: -score[1]) #sort based on score
    # print(cn_data['gem'])
    cn_dataset = {}
    cn_dataset_list = []

    C = C if conceptorProj else np.zeros((wordVecModel.shape[1], wordVecModel.shape[1]))

    for ((word1, word2), score) in dataset:
        #print(word1, word2)
        index1 = vocab.index(word1)
        index2 = vocab.index(word2)
        sim_score = 1 - cosine_similarity([wordVecModel[index1]  - (C @ wordVecModel[index1])], [wordVecModel[index2] - (C @ wordVecModel[index2])])[0][0]
        cn_dataset[(word1, word2)] = sim_score
        cn_dataset_list.append(((word1, word2),sim_score))
    cn_dataset_list.sort(key = lambda score: score[1])
    spearman_list1=[]
    spearman_list2=[]
    for pos_1, (pair, score_1) in enumerate(dataset):
        score_2 = cn_dataset[pair]
        pos_2 = cn_dataset_list.index((pair, score_2))
        spearman_list1.append(pos_1)
        spearman_list2.append(pos_2)
    rho = spearmanr(spearman_list1, spearman_list2)
    return rho[0]


def similarity_eval_sparse(dataSetAddress, wordVecModel, vocab):

    if type(wordVecModel) != csr_matrix:
        wordVecModel = csr_matrix(wordVecModel.tocoo())

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


    for (x, y) in pair_list:
        (word_i, word_j) = x

        #print(wordVecModel[D_vocab_index[word_i]].todense().shape)

        if norm(wordVecModel[D_vocab_index[word_i]].todense()) == 0 or \
            norm(wordVecModel[D_vocab_index[word_j]].todense()) == 0:
            current_distance = 0
        else:
            current_distance = cosine(wordVecModel[D_vocab_index[word_i]].todense(),
                                      wordVecModel[D_vocab_index[word_j]].todense())
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


def evaluation_wordvectorsorg(D_embedding_vec, folder_evaluation, L_fname,
                              is_read = True,
                              is_write_conceptored_embedding = False,
                              folder_write = "",
                              folder_data = "",
                              fname_wordlist = "",
                              D_embedding_fname = None,
                              is_tmp_perturb = False,
                              is_read_dense = True,
                              is_appended_score_writing = False):

    D_embedding_abbt_result = dict()
    D_embedding_abbt_result["word2vec"], D_embedding_abbt_result["glove840B300D"] = load_abtt_results()

    D_embedding_beta = {"word2vec": 2, "glove840B300D": 1, "fasttextCrawl": 1,
                        "fasttextWikinews": 1, "smallGlove": 1, "smallW2V": 1,
                        "Nondist300D_wiki200_fullSV": 2, "Nondist300D_wiki200_halfSV": 2,
                        "Nondist300D_fullSV": 2, "Nondist300D_halfSV": 2,
                        "Nondist_300D_hashing": 1}
    D_embedding_alpha = {"fasttextCrawl": 1, "word2vec": 1, "glove840B300D": 2,
                         "fasttextWikinews": 2, "smallGlove": 2, "smallW2V": 2}
    D_embedding_Cproto = dict()
    D_embedding_Cadjusted = dict()

    wordSimResult = {}

    for embedding in D_embedding_vec:

        if not is_read:
            print("\tnow reading embedding % s" % embedding)
            if not exists(path_join(folder_data, D_embedding_fname[embedding])):
                print("Error: no embedding file of % s: % s. Skipped." % (embedding, D_embedding_fname[embedding]))
                continue
            tmp_D_embedding_vec = read_nondist_wordvec(folder_data = folder_data,
                                                       fname_wordlist = fname_wordlist,
                                                       D_embedding_fname = {embedding: D_embedding_fname[embedding]},
                                                       is_read_dense = is_read_dense)
            tmp_wordVecModel =  tmp_D_embedding_vec[embedding][1]
            tmp_vocab =  tmp_D_embedding_vec[embedding][0]

        else:
            print("\tembedding % s has been read already" % embedding)
            tmp_wordVecModel = D_embedding_vec[embedding][1]
            tmp_vocab = D_embedding_vec[embedding][0]

        if is_tmp_perturb:
            tmp_wordVecModel = perturb_magnitude(tmp_wordVecModel)
            embedding = embedding + "_" + str(randint(low=0, high=10000))

        D_embedding_Cproto[embedding] = proto_conceptor(tmp_wordVecModel, embedding, alpha=D_embedding_alpha.get(embedding, 1))
        if embedding not in D_embedding_beta:
            if trace(D_embedding_Cproto[embedding]) / D_embedding_Cproto[embedding].shape[0] < 0.1:
                D_embedding_beta[embedding] = 2
            else:
                D_embedding_beta[embedding] = 1
            print("\tWarning: missing beta value of % s, using beta = % s" % (embedding, D_embedding_beta[embedding]))
        D_embedding_Cadjusted[embedding] = PHI(D_embedding_Cproto[embedding], D_embedding_beta[embedding])

        print('\tbeta of % s = % s, alpha = % s' % (embedding, D_embedding_beta[embedding], D_embedding_alpha.get(embedding, 1)))
        print('\tQuota for % s conceptor is' % embedding, trace(D_embedding_Cadjusted[embedding]) / D_embedding_Cproto[embedding].shape[0])
        print('\tQuota for % s conceptor (before PHI) is' % embedding, trace(D_embedding_Cproto[embedding]) / D_embedding_Cproto[embedding].shape[0])

        if is_write_conceptored_embedding:
            with open(path_join(folder_write, "% s+conceptored.txt" % embedding), "w") as fw:
                fw.write("% s % s\n" % tmp_wordVecModel.shape)
                for tmp_word, line in zip(tmp_vocab, tmp_wordVecModel - tmp_wordVecModel @ D_embedding_Cadjusted[embedding]):
                    fw.write(tmp_word + " " + " ".join(map(str, line)) + "\n")
            print("\tfinished writing embedding % s" % embedding)

        for fname in L_fname:
            dataSetAddress = path_join(folder_evaluation, fname)
            print('\tevaluating the data set', fname)

            tmp_similarity_conceptor = similarity_eval_tl(dataSetAddress,
                                             wordVecModel = tmp_wordVecModel,
                                             vocab = tmp_vocab,
                                             conceptorProj = True,
                                             C = D_embedding_Cadjusted[embedding])
            tmp_similarity_raw = similarity_eval_tl(dataSetAddress,
                                                 wordVecModel=tmp_wordVecModel,
                                                 vocab=tmp_vocab,
                                             conceptorProj = False,
                                             C = D_embedding_Cadjusted[embedding])

            tmp_similarity_abtt1 = similarity_eval_tl(dataSetAddress,
                                                 wordVecModel= abtt(tmp_wordVecModel, D = 1),
                                                 vocab=tmp_vocab,
                                                 conceptorProj=False,
                                                 C=D_embedding_Cadjusted[embedding])

            tmp_similarity_abtt2 = similarity_eval_tl(dataSetAddress,
                                                   wordVecModel=abtt(tmp_wordVecModel, D=2),
                                                   vocab=tmp_vocab,
                                                   conceptorProj=False,
                                                   C=D_embedding_Cadjusted[embedding])

            tmp_similarity_abtt3 = similarity_eval_tl(dataSetAddress,
                                                   wordVecModel=abtt(tmp_wordVecModel, D=3),
                                                   vocab=tmp_vocab,
                                                   conceptorProj=False,
                                                   C=D_embedding_Cadjusted[embedding])

            tmp_similarity_abtt4 = similarity_eval_tl(dataSetAddress,
                                                   wordVecModel=abtt(tmp_wordVecModel, D=4),
                                                   vocab=tmp_vocab,
                                                   conceptorProj=False,
                                                   C=D_embedding_Cadjusted[embedding])

            tmp_similarity_abtt5 = similarity_eval_tl(dataSetAddress,
                                                   wordVecModel=abtt(tmp_wordVecModel, D=5),
                                                   vocab=tmp_vocab,
                                                   conceptorProj=False,
                                                   C=D_embedding_Cadjusted[embedding])

            if embedding in D_embedding_abbt_result:
                print('\t% s + ABTT: %.4f ' % (embedding, D_embedding_abbt_result.get(embedding, dict()).get(fname, 0)))
            print('\t% s + conceptor: %.4f' % (embedding, tmp_similarity_conceptor))
            print('\t% s + raw: %.4f' % (embedding, tmp_similarity_raw))
            print('\t% s + abtt1: %.4f' % (embedding, tmp_similarity_abtt1))
            print('\t% s + abtt2: %.4f' % (embedding, tmp_similarity_abtt2))
            print('\t% s + abtt3: %.4f' % (embedding, tmp_similarity_abtt3))
            print('\t% s + abtt4: %.4f' % (embedding, tmp_similarity_abtt4))
            print('\t% s + abtt5: %.4f' % (embedding, tmp_similarity_abtt5))

            wordSimResult["% s+% s" % (embedding, fname.split(".")[0])] = \
                    [embedding,
                     fname.split(".")[0],
                     round(tmp_similarity_conceptor * 100, 2),
                     round(tmp_similarity_raw * 100, 2),
                     round(tmp_similarity_abtt1 * 100, 2),
                     round(tmp_similarity_abtt2 * 100, 2),
                     round(tmp_similarity_abtt3 * 100, 2),
                     round(tmp_similarity_abtt4 * 100, 2),
                     round(tmp_similarity_abtt5 * 100, 2),]

        print('\n')

    for embedding in D_embedding_Cproto:
        print('beta of % s = % s, alpha = % s' % (embedding, D_embedding_beta[embedding], D_embedding_alpha.get(embedding, 1)))
        print('Quota for % s conceptor is' % embedding, trace(D_embedding_Cadjusted[embedding]) / D_embedding_Cproto[embedding].shape[0])
        print('Quota for % s conceptor (before PHI) is' % embedding, trace(D_embedding_Cproto[embedding]) / D_embedding_Cproto[embedding].shape[0])
    print()

    wordSimResult_df = pd.DataFrame(wordSimResult, index=['emb', 'task', 'conceptor', 'raw', "abtt1", "abtt2", "abtt3", "abtt4", "abtt5"]).T
    if is_appended_score_writing:
        if exists(path_join(os.getcwd(), "wordSimResult_df.csv")):
            wordSimResult_df = pd.concat([pd.read_csv(path_join(os.getcwd(), "wordSimResult_df.csv"), index_col=0), wordSimResult_df])
    wordSimResult_df.to_csv(path_join(os.getcwd(), "wordSimResult_df.csv"))

    with open("quota.txt", "w") as fw:
        fw.write("\t".join(["embedding", "beta", "alpha", "quota", "quota_beforePHI"]) + "\n")
        for embedding in D_embedding_Cproto:
            #fw.write('beta of % s = % s, alpha = % s\n' % (embedding, D_embedding_beta[embedding], D_embedding_alpha.get(embedding, 1)))
            #fw.write('Quota for % s conceptor is % s\n' % (embedding, trace(D_embedding_Cadjusted[embedding]) / D_embedding_Cproto[embedding].shape[0]))
            #fw.write('Quota for % s conceptor (before PHI) is % s\n' % (embedding, trace(D_embedding_Cproto[embedding]) / D_embedding_Cproto[embedding].shape[0]))
            fw.write("\t".join([embedding, str(D_embedding_beta[embedding]), str(D_embedding_alpha.get(embedding, 1)),
                                str(trace(D_embedding_Cadjusted[embedding]) / D_embedding_Cproto[embedding].shape[0]),
                                str(trace(D_embedding_Cproto[embedding]) / D_embedding_Cproto[embedding].shape[0])]) + "\n")

    #ax = wordSimResult_df.plot(kind="bar")
    #ax.legend(loc="best")
    #ax.set_ylim(20, 100)
    #ax.set_ylabel("Pearson correlation coefficient x 100")
    #plt.show()

    return


def calculate_purity(y_true, y_pred):
    """
    Calculate purity for given true and predicted cluster labels.
    Parameters
    ----------
    y_true: array, shape: (n_samples, 1)
      True cluster labels
    y_pred: array, shape: (n_samples, 1)
      Cluster assingment.
    Returns
    -------
    purity: float
      Calculated purity.
    """
    assert len(y_true) == len(y_pred)
    true_clusters = np.zeros(shape=(len(set(y_true)), len(y_true)))
    pred_clusters = np.zeros_like(true_clusters)
    for id, cl in enumerate(set(y_true)):
        true_clusters[id] = (y_true == cl).astype("int")
    for id, cl in enumerate(set(y_pred)):
        pred_clusters[id] = (y_pred == cl).astype("int")

    M = pred_clusters.dot(true_clusters.T)
    return 1. / len(y_true) * np.sum(np.max(M, axis=1))


def categorization_eval(categorizationFile, wordVecModel, vocab, conceptorProj=False, C = None, method='fixed'):

    C = C if conceptorProj else np.zeros((wordVecModel.shape[1], wordVecModel.shape[1]))

    wordVectorsMat = wordVecModel - wordVecModel @ C
    modelVocab = vocab

    categorty_list = []
    word_list = []

    with open(categorizationFile, newline='') as csvfile:
        next(csvfile)
        reader = csv.reader(csvfile, quotechar='|')
        for row in reader:
            if len(row[2]) != 0 and row[2] in modelVocab:
                categorty_list.append(row[1])
                word_list.append(row[2])

    wordVectorsMat = np.array([wordVectorsMat[modelVocab.index(word)] for word in word_list])

    initCentroids = []
    for category in set(categorty_list):
        indicesCategory = [i for i in range(len(categorty_list)) if categorty_list[i] == category]
        initCentroid = np.mean(wordVectorsMat[indicesCategory, :], axis=0)
        initCentroids.append(initCentroid)

    initCentroids = np.array(initCentroids)

    if method == 'fixed':

        predClusters = KMeans(init=initCentroids, n_clusters=len(set(categorty_list))).fit_predict(wordVectorsMat)
        purity = calculate_purity(np.array(categorty_list), predClusters)

    else:

        predClusters = KMeans(n_init=10000, n_clusters=len(set(categorty_list))).fit_predict(wordVectorsMat)
        purity = calculate_purity(np.array(categorty_list), predClusters)

    return purity


def evaluation_conceptCategorization(D_embedding_vec, folder_evaluation, L_fname,
                              is_read = True,
                              is_write_conceptored_embedding = False,
                              folder_write = "",
                              folder_data = "",
                              fname_wordlist = "",
                              D_embedding_fname = None,
                              is_tmp_perturb = False,
                              is_read_dense = True,
                              is_appended_score_writing = False):

    D_embedding_abbt_result = dict()
    D_embedding_abbt_result["word2vec"], D_embedding_abbt_result["glove840B300D"] = load_abtt_results()

    D_embedding_beta = {"word2vec": 2, "glove840B300D": 1, "fasttextCrawl": 1,
                        "Nondist300D_wiki200_fullSV": 2, "Nondist300D_wiki200_halfSV": 2,
                        "Nondist300D_fullSV": 2, "Nondist300D_halfSV": 2,
                        "Nondist_300D_hashing": 1}
    D_embedding_alpha = {"fasttextCrawl": 1, "word2vec": 1, "glove840B300D": 2}
    D_embedding_Cproto = dict()
    D_embedding_Cadjusted = dict()

    wordSimResult = {}

    for embedding in D_embedding_vec:

        if not is_read:
            print("\tnow reading embedding % s" % embedding)
            if not exists(path_join(folder_data, D_embedding_fname[embedding])):
                print("Error: no embedding file of % s: % s. Skipped." % (embedding, D_embedding_fname[embedding]))
                continue
            tmp_D_embedding_vec = read_nondist_wordvec(folder_data = folder_data,
                                                       fname_wordlist = fname_wordlist,
                                                       D_embedding_fname = {embedding: D_embedding_fname[embedding]},
                                                       is_read_dense = is_read_dense)
            tmp_wordVecModel =  tmp_D_embedding_vec[embedding][1]
            tmp_vocab =  tmp_D_embedding_vec[embedding][0]

        else:
            print("\tembedding % s has been read already" % embedding)
            tmp_wordVecModel = D_embedding_vec[embedding][1]
            tmp_vocab = D_embedding_vec[embedding][0]

        if is_tmp_perturb:
            tmp_wordVecModel = perturb_magnitude(tmp_wordVecModel)
            embedding = embedding + "_" + str(randint(low=0, high=10000))

        D_embedding_Cproto[embedding] = proto_conceptor(tmp_wordVecModel, embedding, alpha=D_embedding_alpha.get(embedding, 1))
        if embedding not in D_embedding_beta:
            if trace(D_embedding_Cproto[embedding]) / D_embedding_Cproto[embedding].shape[0] < 0.1:
                D_embedding_beta[embedding] = 2
            else:
                D_embedding_beta[embedding] = 1
            print("\tWarning: missing beta value of % s, using beta = % s" % (embedding, D_embedding_beta[embedding]))
        D_embedding_Cadjusted[embedding] = PHI(D_embedding_Cproto[embedding], D_embedding_beta[embedding])

        print('\tbeta of % s = % s, alpha = % s' % (embedding, D_embedding_beta[embedding], D_embedding_alpha.get(embedding, 1)))
        print('\tQuota for % s conceptor is' % embedding, trace(D_embedding_Cadjusted[embedding]) / D_embedding_Cproto[embedding].shape[0])
        print('\tQuota for % s conceptor (before PHI) is' % embedding, trace(D_embedding_Cproto[embedding]) / D_embedding_Cproto[embedding].shape[0])

        if is_write_conceptored_embedding:
            with open(path_join(folder_write, "% s+conceptored.txt" % embedding), "w") as fw:
                fw.write("% s % s\n" % tmp_wordVecModel.shape)
                for tmp_word, line in zip(tmp_vocab, tmp_wordVecModel - tmp_wordVecModel @ D_embedding_Cadjusted[embedding]):
                    fw.write(tmp_word + " " + " ".join(map(str, line)) + "\n")
            print("\tfinished writing embedding % s" % embedding)

        for fname in L_fname:
            dataSetAddress = path_join(folder_evaluation, fname)
            print('\tevaluating the data set', fname)

            tmp_categorization_conceptor = categorization_eval(dataSetAddress,
                                             wordVecModel = tmp_wordVecModel,
                                             vocab = tmp_vocab,
                                             conceptorProj = True,
                                             C = D_embedding_Cadjusted[embedding])
            tmp_categorization_raw = categorization_eval(dataSetAddress,
                                                 wordVecModel=tmp_wordVecModel,
                                                 vocab=tmp_vocab,
                                             conceptorProj = False,
                                             C = D_embedding_Cadjusted[embedding])

            tmp_categorization_abtt1 = categorization_eval(dataSetAddress,
                                                 wordVecModel= abtt(tmp_wordVecModel, D = 1),
                                                 vocab=tmp_vocab,
                                                 conceptorProj=False,
                                                 C=D_embedding_Cadjusted[embedding])

            tmp_categorization_abtt2 = categorization_eval(dataSetAddress,
                                                   wordVecModel=abtt(tmp_wordVecModel, D=2),
                                                   vocab=tmp_vocab,
                                                   conceptorProj=False,
                                                   C=D_embedding_Cadjusted[embedding])

            tmp_categorization_abtt3 = categorization_eval(dataSetAddress,
                                                   wordVecModel=abtt(tmp_wordVecModel, D=3),
                                                   vocab=tmp_vocab,
                                                   conceptorProj=False,
                                                   C=D_embedding_Cadjusted[embedding])

            tmp_categorization_abtt4 = categorization_eval(dataSetAddress,
                                                   wordVecModel=abtt(tmp_wordVecModel, D=4),
                                                   vocab=tmp_vocab,
                                                   conceptorProj=False,
                                                   C=D_embedding_Cadjusted[embedding])

            tmp_categorization_abtt5 = categorization_eval(dataSetAddress,
                                                   wordVecModel=abtt(tmp_wordVecModel, D=5),
                                                   vocab=tmp_vocab,
                                                   conceptorProj=False,
                                                   C=D_embedding_Cadjusted[embedding])

            print('\t% s + conceptor: %.4f' % (embedding, tmp_categorization_conceptor))
            print('\t% s + raw: %.4f' % (embedding, tmp_categorization_raw))
            print('\t% s + abtt1: %.4f' % (embedding, tmp_categorization_abtt1))
            print('\t% s + abtt2: %.4f' % (embedding, tmp_categorization_abtt2))
            print('\t% s + abtt3: %.4f' % (embedding, tmp_categorization_abtt3))
            print('\t% s + abtt4: %.4f' % (embedding, tmp_categorization_abtt4))
            print('\t% s + abtt5: %.4f' % (embedding, tmp_categorization_abtt5))

            wordSimResult["% s+% s" % (embedding, fname.split(".")[0])] = \
                    [embedding,
                     fname.split(".")[0],
                     round(tmp_categorization_conceptor * 100, 2),
                     round(tmp_categorization_raw * 100, 2),
                     round(tmp_categorization_abtt1 * 100, 2),
                     round(tmp_categorization_abtt2 * 100, 2),
                     round(tmp_categorization_abtt3 * 100, 2),
                     round(tmp_categorization_abtt4 * 100, 2),
                     round(tmp_categorization_abtt5 * 100, 2),]

        print('\n')

    for embedding in D_embedding_Cproto:
        print('beta of % s = % s, alpha = % s' % (embedding, D_embedding_beta[embedding], D_embedding_alpha.get(embedding, 1)))
        print('Quota for % s conceptor is' % embedding, trace(D_embedding_Cadjusted[embedding]) / D_embedding_Cproto[embedding].shape[0])
        print('Quota for % s conceptor (before PHI) is' % embedding, trace(D_embedding_Cproto[embedding]) / D_embedding_Cproto[embedding].shape[0])
    print()

    wordSimResult_df = pd.DataFrame(wordSimResult, index=['emb', 'task', 'conceptor', 'raw', "abtt1", "abtt2", "abtt3", "abtt4", "abtt5"]).T
    if is_appended_score_writing:
        if exists(path_join(os.getcwd(), "wordCategorizationResult_df.csv")):
            wordSimResult_df = pd.concat([pd.read_csv(path_join(os.getcwd(), "wordCategorizationResult_df.csv"), index_col=0), wordSimResult_df])
    wordSimResult_df.to_csv(path_join(os.getcwd(), "wordCategorizationResult_df.csv"))

    with open("quota.txt", "w") as fw:
        fw.write("\t".join(["embedding", "beta", "alpha", "quota", "quota_beforePHI"]) + "\n")
        for embedding in D_embedding_Cproto:
            #fw.write('beta of % s = % s, alpha = % s\n' % (embedding, D_embedding_beta[embedding], D_embedding_alpha.get(embedding, 1)))
            #fw.write('Quota for % s conceptor is % s\n' % (embedding, trace(D_embedding_Cadjusted[embedding]) / D_embedding_Cproto[embedding].shape[0]))
            #fw.write('Quota for % s conceptor (before PHI) is % s\n' % (embedding, trace(D_embedding_Cproto[embedding]) / D_embedding_Cproto[embedding].shape[0]))
            fw.write("\t".join([embedding, str(D_embedding_beta[embedding]), str(D_embedding_alpha.get(embedding, 1)),
                                str(trace(D_embedding_Cadjusted[embedding]) / D_embedding_Cproto[embedding].shape[0]),
                                str(trace(D_embedding_Cproto[embedding]) / D_embedding_Cproto[embedding].shape[0])]) + "\n")

    #ax = wordSimResult_df.plot(kind="bar")
    #ax.legend(loc="best")
    #ax.set_ylim(20, 100)
    #ax.set_ylabel("Pearson correlation coefficient x 100")
    #plt.show()

    return


def evaluation_fullsparsenondist_wordvectorsorg(D_embedding_vec, folder_evaluation, L_fname):

    wordSimResult = {}

    for embedding in D_embedding_vec:

        print("\tembedding % s has been read already" % embedding)
        tmp_wordVecModel = D_embedding_vec[embedding][1]
        tmp_vocab = D_embedding_vec[embedding][0]

        for fname in L_fname:
            dataSetAddress = path_join(folder_evaluation, fname)
            print('\tevaluating the data set', fname)

            tmp_similarity = similarity_eval_sparse(dataSetAddress,
                                             wordVecModel = tmp_wordVecModel,
                                             vocab = tmp_vocab)

            print('\t% s + fullsparse: %.4f' % (embedding, tmp_similarity))

            wordSimResult["% s+% s" % (embedding, fname.split(".")[0])] = \
                    [round(tmp_similarity * 100, 2)]

        print('\n')

    wordSimResult_df = pd.DataFrame(wordSimResult, index=["sparse"]).T
    wordSimResult_df.to_csv(path_join(os.getcwd(), "wordSimResult_fullsparseNondist_df.csv"))

    return


def Main():

    print("start running conceptor on nondistributional word vectors")
    now_time = datetime.datetime.now
    start_time = now_time()
    print()

    '''
    D_embedding_vec = dict()
    print("\tstart reading distributional embeddings", (now_time() - start_time).seconds)
    D_embedding_vec = read_dist_wordvec({"word2vec": FNAME_WORD2VEC_GOOGLENEWS,
                                         "glove840B300D": FNAME_GLOVE_840B300D,
                                         "fasttext2": path_join(FOLDER_DISTRIBUTIONALVEC, "fasttext.bin"),
                                         "glove6B300D": path_join(FOLDER_DISTRIBUTIONALVEC, "Glove6B300D.txt"),
                                         "fasttextCrawl": FNAME_FASTTEXT_CRAWL,
                                         "fasttextWikinews": FNAME_FASTTEXT_WIKINEWS,
                                         "smallGlove": path_join(FOLDER_DISTRIBUTIONALVEC, "small_glove.txt"),
                                         "smallW2V": path_join(FOLDER_DISTRIBUTIONALVEC, "small_word2vec.txt"),
                                         },
                                        FNAME_WIKIWORDS)
    print("\tfinished reading distributional embeddings", (now_time() - start_time).seconds)
    print()
    evaluation_wordvectorsorg(D_embedding_vec=D_embedding_vec,
                              folder_evaluation=FOLDER_EVALUATION_WORDVECTORSORG,
                              L_fname=LIST_FNAME_EVALUATION_WORDVECTORSORG,
                              is_read=True,
                              is_write_conceptored_embedding=True,
                              folder_write=FOLDER_RES)
    '''

    '''
    D_embedding_vec = dict()
    print("\tstart reading nondistributional embeddings", (now_time() - start_time).seconds)
    D_embedding_vec_nondist = read_nondist_wordvec(path_join(os.getcwd(), "data\\NondistributionalVec"),
                                                    FNAME_WIKIWORDS,
                                                   {"Nondist_300D_fullSV": "Nondist_300D_noSV.txt"},
                                                    is_read_dense=True)
    print("\tfinished reading nondistributional embeddings", (now_time() - start_time).seconds)
    D_embedding_vec.update(D_embedding_vec_nondist)
    print("data reading finished", (now_time() - start_time).seconds)
    print()
    evaluation_wordvectorsorg(D_embedding_vec=D_embedding_vec,
                              folder_evaluation=FOLDER_EVALUATION_WORDVECTORSORG,
                              L_fname=LIST_FNAME_EVALUATION_WORDVECTORSORG,
                              is_read=True,
                              is_write_conceptored_embedding=True,
                              folder_write=FOLDER_RES)
    '''

    '''
    evaluation_wordvectorsorg(D_embedding_vec = DICT_NONDIST_EMBEDDING_FNAME,
                              folder_evaluation = FOLDER_EVALUATION_WORDVECTORSORG,
                              L_fname = LIST_FNAME_EVALUATION_WORDVECTORSORG,
                              is_read = False,
                              is_write_conceptored_embedding = True,
                              folder_write = FOLDER_RES,
                              folder_data = FOLDER_NONDISTRIBUTIONALVEC,
                              fname_wordlist = FNAME_WIKIWORDS,
                              D_embedding_fname = DICT_NONDIST_EMBEDDING_FNAME,
                              is_read_dense = True)
    '''

    '''
    print("\tstart reading sparse nondistributional embeddings", (now_time() - start_time).seconds)
    D_embedding_vec = read_nondist_wordvec(FOLDER_NONDISTRIBUTIONALVEC,
                                           FNAME_WIKIWORDS,
                                           dict(),
                                           is_read_dense=False)
    print("\tfinished reading nondistributional embeddings", (now_time() - start_time).seconds)
    print("data reading finished", (now_time() - start_time).seconds)
    print()
    evaluation_fullsparsenondist_wordvectorsorg(D_embedding_vec,
                                                folder_evaluation = FOLDER_EVALUATION_WORDVECTORSORG,
                                                L_fname=LIST_FNAME_EVALUATION_WORDVECTORSORG)
    '''

    '''
    evaluation_wordvectorsorg(D_embedding_vec=DICT_ALIGNED_NONDIST_EMBEDDING_FNAME,
                              folder_evaluation=FOLDER_EVALUATION_WORDVECTORSORG,
                              L_fname=LIST_FNAME_EVALUATION_WORDVECTORSORG,
                              is_read=False,
                              is_write_conceptored_embedding=True,
                              folder_write=FOLDER_RES,
                              folder_data=FOLDER_ALIGNED_NONDISTRIBUTIONALVEC,
                              fname_wordlist=FNAME_WIKIWORDS,
                              D_embedding_fname=DICT_ALIGNED_NONDIST_EMBEDDING_FNAME,
                              is_read_dense=True)
    '''


    '''
    evaluation_wordvectorsorg(#D_embedding_vec={"Nondist_300D_hashingBiased": "Nondist_300D_hashingBiased.txt"},
                              D_embedding_vec={"Nondist_300D_hashing": "Nondist_300D_hashing.txt"},
                              folder_evaluation=FOLDER_EVALUATION_WORDVECTORSORG,
                              L_fname=LIST_FNAME_EVALUATION_WORDVECTORSORG,
                              is_read=False,
                              is_write_conceptored_embedding=True,
                              folder_write=FOLDER_RES,
                              folder_data=FOLDER_NONDISTRIBUTIONALVEC,
                              fname_wordlist=FNAME_WIKIWORDS,
                              #D_embedding_fname={"Nondist_300D_hashingBias": "Nondist_300D_hashingBiased.txt"},
                              D_embedding_fname={"Nondist_300D_hashing": "Nondist_300D_hashing.txt"},
                              is_read_dense=True)
    '''

    '''
    D_embedding_vec_my = {"word2vec_all": "word2vec_all.txt",
                          "word2vec_merged32": "word2vec_merged32.txt",
                          "word2vec_conceptor_merged32": "word2vec_conceptor_merged32.txt",
                          "word2vec0": "word2vec0.txt",
                          "word2vec1": "word2vec1.txt",
                          "word2vec2": "word2vec2.txt",
                          "word2vec3": "word2vec3.txt",
                          "word2vec4": "word2vec4.txt",
                          "word2vec5": "word2vec5.txt",
                          "word2vec6": "word2vec6.txt",
                          "word2vec7": "word2vec7.txt",
                          "word2vec_bs_0" : "word2vec_bs_0.txt",
                          "fasttext_all": "fasttext_all.txt",
                          "fasttext_merged8": "fasttext_merged8.txt",
                          "fasttext_conceptor_merged8": "fasttext_conceptor_merged8.txt",
                          "fasttext0": "fasttext0.txt",
                          "fasttext1": "fasttext1.txt",
                          "fasttext2": "fasttext2.txt",
                          "fasttext3": "fasttext3.txt",
                          }
    evaluation_wordvectorsorg(
        D_embedding_vec=D_embedding_vec_my,
        folder_evaluation=FOLDER_EVALUATION_WORDVECTORSORG,
        L_fname=LIST_FNAME_EVALUATION_WORDVECTORSORG,
        is_read=False,
        is_write_conceptored_embedding=True,
        folder_write=FOLDER_RES,
        folder_data=FOLDER_MYVEC,
        fname_wordlist=FNAME_WIKIWORDS,
        D_embedding_fname=D_embedding_vec_my,
        is_read_dense=True)
    '''

    '''
    folder_permutationAlign_vec = r"D:\Work\JRodu\ConceptorOnNondist\MUSE\MUSE-master\res_myAlign_permutation"
    D_embedding_vec_my = dict()
    for fname in os.listdir(folder_permutationAlign_vec):
        D_embedding_vec_my[fname.split(".txt")[0]] = fname
    evaluation_wordvectorsorg(
        D_embedding_vec=D_embedding_vec_my,
        folder_evaluation=FOLDER_EVALUATION_WORDVECTORSORG,
        L_fname=LIST_FNAME_EVALUATION_WORDVECTORSORG,
        is_read=False,
        is_write_conceptored_embedding=False,
        folder_write=FOLDER_RES,
        folder_data=folder_permutationAlign_vec,
        fname_wordlist=FNAME_WIKIWORDS,
        D_embedding_fname=D_embedding_vec_my,
        is_read_dense=True)
    '''

    '''
    folder_permutationAlign_vec = r"D:\Work\JRodu\ConceptorOnNondist\MUSE\MUSE-master\res_myAlign_permutationSource"
    D_embedding_vec_my = dict()
    for fname in os.listdir(folder_permutationAlign_vec):
        D_embedding_vec_my[fname.split(".txt")[0]] = fname
    evaluation_wordvectorsorg(
        D_embedding_vec=D_embedding_vec_my,
        folder_evaluation=FOLDER_EVALUATION_WORDVECTORSORG,
        L_fname=LIST_FNAME_EVALUATION_WORDVECTORSORG,
        is_read=False,
        is_write_conceptored_embedding=False,
        folder_write=FOLDER_RES,
        folder_data=folder_permutationAlign_vec,
        fname_wordlist=FNAME_WIKIWORDS,
        D_embedding_fname=D_embedding_vec_my,
        is_read_dense=True)
    '''

    '''
    D_embedding_vec = dict()
    print("\tstart reading distributional embeddings", (now_time() - start_time).seconds)
    D_embedding_vec = read_dist_wordvec({
                                         "word2vec": FNAME_WORD2VEC_GOOGLENEWS,
                                         #"glove840B300D": FNAME_GLOVE_840B300D,
                                         #"fasttextCrawl": FNAME_FASTTEXT_CRAWL,
                                        },
                                        FNAME_WIKIWORDS)
    print("\tfinished reading distributional embeddings", (now_time() - start_time).seconds)
    print()
    for ii in range(5):
        evaluation_wordvectorsorg(D_embedding_vec=D_embedding_vec,
                                  folder_evaluation=FOLDER_EVALUATION_WORDVECTORSORG,
                                  L_fname=LIST_FNAME_EVALUATION_WORDVECTORSORG,
                                  is_read=True,
                                  is_write_conceptored_embedding=False,
                                  is_tmp_perturb=True,
                                  is_appended_score_writing=True,
                                  )
    '''

    '''
    D_embedding_vec_alignedByVerb = {"fasttextCrawl_by_glove840B300D": "fasttextCrawl_by_glove840B300D.txt",
                                     "fasttextCrawl_by_word2vec": "fasttextCrawl_by_word2vec.txt",
                                     "glove840B300D_by_fasttextCrawl": "glove840B300D_by_fasttextCrawl.txt",
                                     "glove840B300D_by_word2vec": "glove840B300D_by_word2vec.txt",
                                     "word2vec_by_fasttextCrawl": "word2vec_by_fasttextCrawl.txt",
                                     "word2vec_by_glove840B300D": "word2vec_by_glove840B300D.txt",
                                     }
    evaluation_wordvectorsorg(
        D_embedding_vec=D_embedding_vec_alignedByVerb,
        folder_evaluation=FOLDER_EVALUATION_WORDVECTORSORG,
        L_fname=LIST_FNAME_EVALUATION_WORDVECTORSORG,
        is_read=False,
        is_write_conceptored_embedding=False,
        folder_write=FOLDER_RES,
        folder_data=path_join(FOLDER_DATA, "DistAlignByVerb"),
        fname_wordlist=FNAME_WIKIWORDS,
        D_embedding_fname=D_embedding_vec_alignedByVerb,
        is_read_dense=True)
    '''

    '''
    folder_permutationAlign_vec = r"D:\Work\JRodu\ConceptorOnNondist\data\AlignPermuteSelf"
    D_embedding_vec_my = dict()
    for fname in os.listdir(folder_permutationAlign_vec):
        D_embedding_vec_my[fname.split(".txt")[0]] = fname
    evaluation_wordvectorsorg(
        D_embedding_vec=D_embedding_vec_my,
        folder_evaluation=FOLDER_EVALUATION_WORDVECTORSORG,
        L_fname=LIST_FNAME_EVALUATION_WORDVECTORSORG,
        is_read=False,
        is_write_conceptored_embedding=False,
        folder_write=FOLDER_RES,
        folder_data=folder_permutationAlign_vec,
        fname_wordlist=FNAME_WIKIWORDS,
        D_embedding_fname=D_embedding_vec_my,
        is_read_dense=True)
    '''

    '''
    folder_permutationAlign_vec = r"D:\Work\JRodu\ConceptorOnNondist\data\AlignDesignedNoise"
    D_embedding_vec_my = dict()
    for fname in os.listdir(folder_permutationAlign_vec):
        D_embedding_vec_my[fname.split(".txt")[0]] = fname
    evaluation_wordvectorsorg(
        D_embedding_vec=D_embedding_vec_my,
        folder_evaluation=FOLDER_EVALUATION_WORDVECTORSORG,
        L_fname=LIST_FNAME_EVALUATION_WORDVECTORSORG,
        is_read=False,
        is_write_conceptored_embedding=False,
        folder_write=FOLDER_RES,
        folder_data=folder_permutationAlign_vec,
        fname_wordlist=FNAME_WIKIWORDS,
        D_embedding_fname=D_embedding_vec_my,
        is_read_dense=True) 
    '''

    '''
    D_embedding_vec = dict()
    print("\tstart reading distributional embeddings", (now_time() - start_time).seconds)
    D_embedding_vec = read_dist_wordvec({"word2vec": FNAME_WORD2VEC_GOOGLENEWS,
                                         "glove840B300D": FNAME_GLOVE_840B300D,
                                         "fasttextCrawl": FNAME_FASTTEXT_CRAWL},
                                        FNAME_WIKIWORDS)
    print("\tfinished reading distributional embeddings", (now_time() - start_time).seconds)
    print()
    evaluation_conceptCategorization(D_embedding_vec=D_embedding_vec,
                              folder_evaluation=FOLDER_EVALUATION_CATEGORIZATION,
                              L_fname=LIST_FNAME_EVALUATION_CATEGORIZATION,
                              is_read=True)
    '''

    '''
    D_embedding_vec = dict()
    print("\tstart reading nondistributional embeddings", (now_time() - start_time).seconds)
    D_embedding_vec_nondist = read_nondist_wordvec(FOLDER_NONDISTRIBUTIONALVEC,
                                                   FNAME_WIKIWORDS,
                                                   {"Nondist_300D_fullSV": "Nondist_300D_fullSV.txt",
                                                    "Nondist_300D_fullSV_wiki200": "Nondist_300D_fullSV_wiki200.txt",
                                                    "Nondist_300D_fullSV_wiki200freq": "Nondist_300D_fullSV_wiki200freq.txt"},
                                                   is_read_dense=True)
    print("\tfinished reading nondistributional embeddings", (now_time() - start_time).seconds)
    D_embedding_vec.update(D_embedding_vec_nondist)
    print("data reading finished", (now_time() - start_time).seconds)
    print()
    evaluation_conceptCategorization(D_embedding_vec=D_embedding_vec,
                                     folder_evaluation=FOLDER_EVALUATION_CATEGORIZATION,
                                     L_fname=LIST_FNAME_EVALUATION_CATEGORIZATION,
                                     is_read=True,
                                     is_appended_score_writing=True,
                                     )
    '''

    '''
    folder_permutationAlign_vec = r"D:\Work\JRodu\ConceptorOnNondist\data\AlignDesignedNoise"
    D_embedding_vec_my = dict()
    for fname in os.listdir(folder_permutationAlign_vec):
        D_embedding_vec_my[fname.split(".txt")[0]] = fname
    evaluation_conceptCategorization(
        D_embedding_vec=D_embedding_vec_my,
        folder_evaluation=FOLDER_EVALUATION_CATEGORIZATION,
        L_fname=LIST_FNAME_EVALUATION_CATEGORIZATION,
        is_read=False,
        is_write_conceptored_embedding=False,
        folder_write=FOLDER_RES,
        folder_data=folder_permutationAlign_vec,
        fname_wordlist=FNAME_WIKIWORDS,
        D_embedding_fname=D_embedding_vec_my,
        is_read_dense=True,
        #is_appended_score_writing=True,
    )
    '''

    '''
    folder_permutationAlign_vec = r"D:\Work\JRodu\ConceptorOnNondist\data\AlignPermuteSelf"
    D_embedding_vec_my = dict()
    for fname in os.listdir(folder_permutationAlign_vec):
        D_embedding_vec_my[fname.split(".txt")[0]] = fname
    evaluation_conceptCategorization(
        D_embedding_vec=D_embedding_vec_my,
        folder_evaluation=FOLDER_EVALUATION_CATEGORIZATION,
        L_fname=LIST_FNAME_EVALUATION_CATEGORIZATION,
        is_read=False,
        is_write_conceptored_embedding=False,
        folder_write=FOLDER_RES,
        folder_data=folder_permutationAlign_vec,
        fname_wordlist=FNAME_WIKIWORDS,
        D_embedding_fname=D_embedding_vec_my,
        is_read_dense=True,
        is_appended_score_writing=True,
    )
    '''

    '''
    folder_AlignedVec = r"D:\Work\JRodu\ConceptorOnNondist\data\AlignedVecND"
    D_embedding_vec_my = dict()
    for fname in os.listdir(folder_AlignedVec):
        D_embedding_vec_my[fname.split(".txt")[0]] = fname
    evaluation_conceptCategorization(
        D_embedding_vec=D_embedding_vec_my,
        folder_evaluation=FOLDER_EVALUATION_CATEGORIZATION,
        L_fname=LIST_FNAME_EVALUATION_CATEGORIZATION,
        is_read=False,
        folder_data=folder_AlignedVec,
        fname_wordlist=FNAME_WIKIWORDS,
        D_embedding_fname=D_embedding_vec_my,
        is_read_dense=True,
    )
    '''

    '''
    folder_permutationAlign_vec = r"D:\Work\JRodu\ConceptorOnNondist\data\AlignedPermuteVecND"
    D_embedding_vec_my = dict()
    for fname in os.listdir(folder_permutationAlign_vec):
        D_embedding_vec_my[fname.split(".txt")[0]] = fname
    evaluation_conceptCategorization(
        D_embedding_vec=D_embedding_vec_my,
        folder_evaluation=FOLDER_EVALUATION_CATEGORIZATION,
        L_fname=LIST_FNAME_EVALUATION_CATEGORIZATION,
        is_read=False,
        folder_data=folder_permutationAlign_vec,
        fname_wordlist=FNAME_WIKIWORDS,
        D_embedding_fname=D_embedding_vec_my,
        is_read_dense=True,
        is_appended_score_writing=False,
    )
    '''

    '''
    folder_vecto_embedding = r"D:\Work\JRodu\ConceptorOnNondist\data\VectoEmb"
    D_embedding_vec_my = {"DepCbow250D": "DepCbow250D.txt",
                          "DepCbow500D": "DepCbow500D.txt",
                          "DepGlove250D": "DepGlove250D.txt",
                          "DepGlove500D": "DepGlove500D.txt",
                          "DepSg250D": "DepSg250D.txt",
                          "DepSg500D": "DepSg500D.txt",
                          "SVD_BNC": "SVD_BNC.txt",
                          }
    evaluation_conceptCategorization(
        D_embedding_vec=D_embedding_vec_my,
        folder_evaluation=FOLDER_EVALUATION_CATEGORIZATION,
        L_fname=LIST_FNAME_EVALUATION_CATEGORIZATION,
        is_read=False,
        folder_data=folder_vecto_embedding,
        fname_wordlist=FNAME_WIKIWORDS,
        D_embedding_fname=D_embedding_vec_my,
        is_read_dense=True,
        is_write_conceptored_embedding=True,
        folder_write=FOLDER_RES,
    )
    evaluation_wordvectorsorg(
        D_embedding_vec=D_embedding_vec_my,
        folder_evaluation=FOLDER_EVALUATION_WORDVECTORSORG,
        L_fname=LIST_FNAME_EVALUATION_WORDVECTORSORG,
        is_read=False,
        folder_data=folder_vecto_embedding,
        fname_wordlist=FNAME_WIKIWORDS,
        D_embedding_fname=D_embedding_vec_my,
        is_read_dense=True,
    )
    '''

    '''
    folder_eig_embedding = r"D:\Work\JRodu\ConceptorOnNondist\data\Eigenwords"
    D_embedding_vec_my = {"Eigenwords300D": "Eigenwords300D.txt",}
    evaluation_conceptCategorization(
        D_embedding_vec=D_embedding_vec_my,
        folder_evaluation=FOLDER_EVALUATION_CATEGORIZATION,
        L_fname=LIST_FNAME_EVALUATION_CATEGORIZATION,
        is_read=False,
        folder_data=folder_eig_embedding,
        fname_wordlist=FNAME_WIKIWORDS,
        D_embedding_fname=D_embedding_vec_my,
        is_read_dense=True,
        is_write_conceptored_embedding=True,
        folder_write=FOLDER_RES,
    )
    evaluation_wordvectorsorg(
        D_embedding_vec=D_embedding_vec_my,
        folder_evaluation=FOLDER_EVALUATION_WORDVECTORSORG,
        L_fname=LIST_FNAME_EVALUATION_WORDVECTORSORG,
        is_read=False,
        folder_data=folder_eig_embedding,
        fname_wordlist=FNAME_WIKIWORDS,
        D_embedding_fname=D_embedding_vec_my,
        is_read_dense=True,
    )
    '''

    '''
    folder_align_vec = r"D:\Work\JRodu\ConceptorOnNondist\data\AlignedDistByDepSvdbncEig"
    D_embedding_vec_my = dict()
    for fname in os.listdir(folder_align_vec):
        D_embedding_vec_my[fname.split(".txt")[0]] = fname
    evaluation_conceptCategorization(
        D_embedding_vec=D_embedding_vec_my,
        folder_evaluation=FOLDER_EVALUATION_CATEGORIZATION,
        L_fname=LIST_FNAME_EVALUATION_CATEGORIZATION,
        is_read=False,
        folder_data=folder_align_vec,
        fname_wordlist=FNAME_WIKIWORDS,
        D_embedding_fname=D_embedding_vec_my,
        is_read_dense=True,
    )
    evaluation_wordvectorsorg(
        D_embedding_vec=D_embedding_vec_my,
        folder_evaluation=FOLDER_EVALUATION_WORDVECTORSORG,
        L_fname=LIST_FNAME_EVALUATION_WORDVECTORSORG,
        is_read=False,
        folder_data=folder_align_vec,
        fname_wordlist=FNAME_WIKIWORDS,
        D_embedding_fname=D_embedding_vec_my,
        is_read_dense=True,
    )
    '''

    '''
    folder_align_vec = r"D:\Work\JRodu\ConceptorOnNondist\data\AlignedDistByPermuteDepSvdbncEig"
    D_embedding_vec_my = dict()
    for fname in os.listdir(folder_align_vec):
        D_embedding_vec_my[fname.split(".txt")[0]] = fname
    evaluation_conceptCategorization(
        D_embedding_vec=D_embedding_vec_my,
        folder_evaluation=FOLDER_EVALUATION_CATEGORIZATION,
        L_fname=LIST_FNAME_EVALUATION_CATEGORIZATION,
        is_read=False,
        folder_data=folder_align_vec,
        fname_wordlist=FNAME_WIKIWORDS,
        D_embedding_fname=D_embedding_vec_my,
        is_read_dense=True,
    )
    evaluation_wordvectorsorg(
        D_embedding_vec=D_embedding_vec_my,
        folder_evaluation=FOLDER_EVALUATION_WORDVECTORSORG,
        L_fname=LIST_FNAME_EVALUATION_WORDVECTORSORG,
        is_read=False,
        folder_data=folder_align_vec,
        fname_wordlist=FNAME_WIKIWORDS,
        D_embedding_fname=D_embedding_vec_my,
        is_read_dense=True,
    )
    '''

    folder_align_vec = r"D:\Work\JRodu\ConceptorOnNondist\Alignment\res_myAlign"
    D_embedding_vec_my = dict()
    for fname in os.listdir(folder_align_vec):
        D_embedding_vec_my[fname.split(".txt")[0]] = fname
    evaluation_wordvectorsorg(
        D_embedding_vec=D_embedding_vec_my,
        folder_evaluation=FOLDER_EVALUATION_WORDVECTORSORG,
        L_fname=LIST_FNAME_EVALUATION_WORDVECTORSORG,
        is_read=False,
        folder_data=folder_align_vec,
        fname_wordlist=FNAME_WIKIWORDS,
        D_embedding_fname=D_embedding_vec_my,
        is_read_dense=True,
    )


    #for embedding in D_embedding_vec: proto_conceptor(D_embedding_vec[embedding][1], embedding, plotSpectrum=True)

    print()
    print("finished running conceptor on nondistributional word vectors")
    print("total running time =", (now_time() - start_time).seconds)


    return


if __name__ == "__main__":
    Main()