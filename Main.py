

import os
from os.path import join as path_join, exists
import datetime
import gzip


import numpy as np
from numpy import dot, trace, array
from numpy.linalg import svd, norm, inv, eig
import gensim
from gensim.models.keyedvectors import KeyedVectors
import scipy.io as sio
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
import pandas as pd


FOLDER_RES = path_join(os.getcwd(), "res")

FOLDER_DATA = path_join(os.getcwd(), "data")
FOLDER_DISTRIBUTIONALVEC = path_join(FOLDER_DATA, "DistributionalVec")
FNAME_WORD2VEC_GOOGLENEWS = path_join(FOLDER_DISTRIBUTIONALVEC, "GoogleNews-vectors-negative300.bin.gz")
FNAME_GLOVE_840B300D = path_join(FOLDER_DISTRIBUTIONALVEC, "gensim_glove.840B.300d.txt.bin")

#FOLDER_NONDISTRIBUTIONALVEC = path_join(FOLDER_DATA, "NondistributionalVec")
FOLDER_NONDISTRIBUTIONALVEC = r"D:\Work\JRodu\WordEmbeddingDomainSpecific\data source\pretrained embedding\nondist\res"

FNAME_WIKIWORDS = path_join(FOLDER_DATA, "Wiki_vocab_gt200", "enwiki_vocab_min200.txt")

FOLDER_EVALUATION = path_join(FOLDER_DATA, "Evaluate")
FOLDER_EVALUATION_WORDVECTORSORG = path_join(FOLDER_EVALUATION, "wordvectors.org_word-sim")
LIST_FNAME_EVALUATION_WORDVECTORSORG = ['EN-RG-65.txt', 'EN-WS-353-ALL.txt', 'EN-RW-STANFORD.txt',
                                        'EN-MEN-TR-3k.txt', 'EN-MTurk-287.txt', 'EN-SIMLEX-999.txt',
                                        'EN-SimVerb-3500.txt']
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
    with open(fname_wordlist) as ff:
        for line in ff:
            wordlist.add(line.strip().split(" ")[0])

    if is_read_dense:
        ans = dict()
        for embedding in D_embedding_fname:
            vocab, X = [], []
            if not exists(path_join(folder_data, D_embedding_fname[embedding])):
                print("Error: no embedding file of % s: % s. Skipped." % (embedding, D_embedding_fname[embedding]))
            with open(path_join(folder_data, D_embedding_fname[embedding]), encoding = "gbk") as ff:
                for cc, line in enumerate(ff):
                    if cc == 0:
                        tmp = line.strip().split(" ")
                        if len(tmp) == 2 and tmp[0].isdigit() and tmp[1].isdigit():
                            continue
                    line = line.strip().split(" ")
                    if line[0] not in wordlist:
                        continue
                    vocab.append(line[0])
                    X.append(array(line[1: ], dtype = np.float32))
            X = np.array(X)
            ans[embedding] = (vocab, X)

    else:
        ans = dict()
        vocab, data, row_ind, col_ind = [], [], [], []
        with gzip.open(path_join(folder_data, "binary-vectors.txt.gz")) as ff:
            for cc, line in enumerate(ff):
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

        X = csc_matrix((data, (row_ind, col_ind)),
                       shape=(len(vocab), ncol),
                       dtype=float)
        ans["sparse_embedding"] = (vocab, X)

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

    C = C if conceptorProj else np.zeros((wordVecModel.shape[1], wordVecModel.shape[1]))

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


def evaluation_wordvectorsorg(D_embedding_vec, folder_evaluation, L_fname,
                              is_read = True,
                              is_write_conceptored_embedding = False,
                              folder_write = "",
                              folder_data = "",
                              fname_wordlist = "",
                              D_embedding_fname = None,
                              is_read_dense = True):

    D_embedding_abbt_result = dict()
    D_embedding_abbt_result["word2vec"], D_embedding_abbt_result["glove840B300D"] = load_abtt_results()

    D_embedding_beta = {"word2vec": 2, "glove840B300D": 1,
                        "Nondist300D_wiki200_fullSV": 2, "Nondist300D_wiki200_halfSV": 2,
                        "Nondist300D_fullSV": 2, "Nondist300D_halfSV": 2,}
    D_embedding_alpha = {"NondistDense": 10.}
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
            tmp_wordVecModel = D_embedding_vec[embedding][1]
            tmp_vocab = D_embedding_vec[embedding][0]

        D_embedding_Cproto[embedding] = proto_conceptor(tmp_wordVecModel, embedding, alpha=D_embedding_alpha.get(embedding, 1))
        if embedding not in D_embedding_beta:
            if trace(D_embedding_Cproto[embedding]) / D_embedding_Cproto[embedding].shape[0] < 0.1:
                D_embedding_beta[embedding] = 2
            else:
                D_embedding_beta[embedding] = 1
            print("Warning: missing beta value of % s, using beta = % s" % (embedding, D_embedding_beta[embedding]))
        D_embedding_Cadjusted[embedding] = PHI(D_embedding_Cproto[embedding], D_embedding_beta[embedding])

        print('\tbeta of % s = % s, alpha = % s' % (embedding, D_embedding_beta[embedding], D_embedding_alpha.get(embedding, 1)))
        print('\tQuota for % s conceptor is' % embedding, trace(D_embedding_Cadjusted[embedding]) / D_embedding_Cproto[embedding].shape[0])
        print('\tQuota for % s conceptor (before PHI) is' % embedding, trace(D_embedding_Cproto[embedding]) / D_embedding_Cproto[embedding].shape[0])

        if is_write_conceptored_embedding:
            with open(path_join(folder_write, "% s.txt" % embedding), "w") as fw:
                fw.write("% s % s\n" % tmp_wordVecModel.shape)
                for tmp_word, line in zip(tmp_vocab, tmp_wordVecModel - tmp_wordVecModel @ D_embedding_Cadjusted[embedding]):
                    fw.write(tmp_word + " " + " ".join(map(str, line)) + "\n")
            print("\tfinished writing embedding % s" % embedding)

        for fname in L_fname:
            dataSetAddress = path_join(folder_evaluation, fname)
            print('\tevaluating the data set', fname)

            tmp_similarity_conceptor = similarity_eval(dataSetAddress,
                                             wordVecModel = tmp_wordVecModel,
                                             vocab = tmp_vocab,
                                             conceptorProj = True,
                                             C = D_embedding_Cadjusted[embedding])
            tmp_similarity_raw = similarity_eval(dataSetAddress,
                                                 wordVecModel=tmp_wordVecModel,
                                                 vocab=tmp_vocab,
                                             conceptorProj = False,
                                             C = D_embedding_Cadjusted[embedding])

            tmp_similarity_abtt1 = similarity_eval(dataSetAddress,
                                                 wordVecModel= abtt(tmp_wordVecModel, D = 1),
                                                 vocab=tmp_vocab,
                                                 conceptorProj=False,
                                                 C=D_embedding_Cadjusted[embedding])

            tmp_similarity_abtt2 = similarity_eval(dataSetAddress,
                                                   wordVecModel=abtt(tmp_wordVecModel, D=2),
                                                   vocab=tmp_vocab,
                                                   conceptorProj=False,
                                                   C=D_embedding_Cadjusted[embedding])

            tmp_similarity_abtt3 = similarity_eval(dataSetAddress,
                                                   wordVecModel=abtt(tmp_wordVecModel, D=3),
                                                   vocab=tmp_vocab,
                                                   conceptorProj=False,
                                                   C=D_embedding_Cadjusted[embedding])

            tmp_similarity_abtt4 = similarity_eval(dataSetAddress,
                                                   wordVecModel=abtt(tmp_wordVecModel, D=4),
                                                   vocab=tmp_vocab,
                                                   conceptorProj=False,
                                                   C=D_embedding_Cadjusted[embedding])

            tmp_similarity_abtt5 = similarity_eval(dataSetAddress,
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

            if "Nondist" in embedding:
                wordSimResult["% s+% s" % (embedding, fname.split(".")[0])] = \
                    [round(tmp_similarity_conceptor * 100, 2),
                     round(tmp_similarity_raw * 100, 2),
                     round(tmp_similarity_abtt1 * 100, 2),
                     round(tmp_similarity_abtt2 * 100, 2),
                     round(tmp_similarity_abtt3 * 100, 2),
                     round(tmp_similarity_abtt4 * 100, 2),
                     round(tmp_similarity_abtt5 * 100, 2),]

        print('\n')

    for embedding in D_embedding_vec:
        print('beta of % s = % s, alpha = % s' % (embedding, D_embedding_beta[embedding], D_embedding_alpha.get(embedding, 1)))
        print('Quota for % s conceptor is' % embedding, trace(D_embedding_Cadjusted[embedding]) / D_embedding_Cproto[embedding].shape[0])
        print('Quota for % s conceptor (before PHI) is' % embedding, trace(D_embedding_Cproto[embedding]) / D_embedding_Cproto[embedding].shape[0])
    print()

    with open("quota.txt", "w") as fw:
        for embedding in D_embedding_vec:
            fw.write('beta of % s = % s, alpha = % s\n' % (embedding, D_embedding_beta[embedding], D_embedding_alpha.get(embedding, 1)))
            fw.write('Quota for % s conceptor is % s\n' % (embedding, trace(D_embedding_Cadjusted[embedding]) / D_embedding_Cproto[embedding].shape[0]))
            fw.write('Quota for % s conceptor (before PHI) is % s\n' %
                     (embedding, trace(D_embedding_Cproto[embedding]) / D_embedding_Cproto[embedding].shape[0]))

    wordSimResult_df = pd.DataFrame(wordSimResult, index=['conceptor', 'raw', "abtt1", "abtt2", "abtt3", "abtt4", "abtt5"]).T
    wordSimResult_df.to_csv(path_join(os.getcwd(), "wordSimResult_df.csv"))

    ax = wordSimResult_df.plot(kind="bar")
    ax.legend(loc="best")
    ax.set_ylim(20, 100)
    ax.set_ylabel("Pearson correlation coefficient x 100")
    #plt.show()

    return


def Main():

    print("start running conceptor on nondistributional word vectors")
    now_time = datetime.datetime.now
    start_time = now_time()
    print()

    D_embedding_vec = dict()

    '''
    print("\tstart reading distributional embeddings", (now_time() - start_time).seconds)
    D_embedding_vec = read_dist_wordvec({"word2vec": FNAME_WORD2VEC_GOOGLENEWS,
                                         "glove840B300D": FNAME_GLOVE_840B300D},
                                        FNAME_WIKIWORDS)
    print("\tfinished reading distributional embeddings", (now_time() - start_time).seconds)
    '''

    '''
    print("\tstart reading nondistributional embeddings", (now_time() - start_time).seconds)
    D_embedding_vec_nondist = read_nondist_wordvec(FOLDER_NONDISTRIBUTIONALVEC,
                                                           FNAME_WIKIWORDS,
                                                           DICT_NONDIST_EMBEDDING_FNAME,
                                                           is_read_dense=True)
    print("\tfinished reading nondistributional embeddings", (now_time() - start_time).seconds)
    D_embedding_vec.update(D_embedding_vec_nondist)
    print("data reading finished", (now_time() - start_time).seconds)
    print()
    '''

    #for embedding in D_embedding_vec: proto_conceptor(D_embedding_vec[embedding][1], embedding, plotSpectrum = True)

    #evaluation_wordvectorsorg(D_embedding_vec, FOLDER_EVALUATION_WORDVECTORSORG, LIST_FNAME_EVALUATION_WORDVECTORSORG)

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

    print()
    print("finished running conceptor on nondistributional word vectors")
    print("total running time =", (now_time() - start_time).seconds)
    return


if __name__ == "__main__":
    Main()