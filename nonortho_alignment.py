

import os
from os.path import join as path_join, exists
import datetime


import numpy as np
from numpy import array
from numpy.linalg import svd, inv
from scipy.spatial.distance import cosine


def nonorthonormal_procrustes(A, B):
    '''
    http://empslocal.ex.ac.uk/people/staff/reverson/uploads/Site/procrustes.pdf
    Fit A = B @ W where W^T @ W is diag.
    '''

    return 0.0


def linear_alignment(A, B):
    '''
    Fit A = B @ W with OLS. Regard A as Y and B as X, we have W = (X^T X)^(-1) X^T Y.
    '''
    #print(inv(B.transpose() @ B).shape, B.shape, A.shape)
    W = inv(B.transpose() @ B) @ B.transpose() @ A
    #print(W.shape)
    return W


def align_two_file(fname_nondist, fname_dist, fname_wordlist, folder_write):

    wordlist = set()
    with open(fname_wordlist) as f_wordlist:
        for line in f_wordlist:
            wordlist.add(line.strip().split(" ")[0])

    D_embedding_fname = {"nondist": fname_nondist, "dist": fname_dist}
    D_embedding_vec = {}
    for embedding in D_embedding_fname:
        vocab, X = [], []
        with open(path_join(D_embedding_fname[embedding]), encoding="gbk") as f_embedding:
            for cc, line in enumerate(f_embedding):
                if cc == 0:
                    tmp = line.strip().split(" ")
                    if len(tmp) == 2 and tmp[0].isdigit() and tmp[1].isdigit():
                        continue
                line = line.strip().split(" ")
                if line[0] not in wordlist:
                    continue
                vocab.append(line[0])
                X.append(array(line[1:], dtype=np.float32))
        X = array(X)
        D_embedding_vec[embedding] = (vocab, X)

    vocab_nondist, X_nondist = D_embedding_vec["nondist"]
    vocab_dist, X_dist = D_embedding_vec["dist"]

    word_used = list(set.intersection(set(vocab_nondist), set(vocab_dist)))

    A = X_nondist[[vocab_nondist.index(i) for i in word_used], :]
    B = X_dist[[vocab_dist.index(i) for i in word_used], :]

    W = linear_alignment(A, B)
    X_dist_aligned = X_dist @ W
    #print(X_dist_aligned[:, :10])

    fname_write = "% s-by-% s.txt" % (os.path.split(fname_dist)[1].split(".txt")[0], os.path.split(fname_nondist)[1].split(".txt")[0])
    with open(path_join(folder_write, fname_write), "w") as fw:
        fw.write("% s % s\n" % X_dist_aligned.shape)
        for (tmp_word, tmp_line) in zip(vocab_dist, X_dist_aligned):
            fw.write(tmp_word + " " + " ".join(map(str, tmp_line)) + "\n")

    L_cosine_sim = []
    for word in word_used:
        tmp_cosine_sim = cosine(X_dist[vocab_dist.index(word), :] @ W, X_nondist[vocab_nondist.index(word), :])
        if not np.isnan(tmp_cosine_sim) and -1 <= tmp_cosine_sim <= 1:
            L_cosine_sim.append(1 - tmp_cosine_sim)
        else:
            pass
    #print(len(L_cosine_sim))
    cos_sim = np.mean(L_cosine_sim)
            # L_cosine_sim.append(0)
    # print(len(L_cosine_sim), len(set.intersection(set(vocab_nondist), set(vocab_dist))))
    return cos_sim


FNAME_WORDLIST = r'D:\Work\JRodu\ConceptorOnNondist\data\Wiki_vocab_gt200\enwiki_vocab_min200.txt'
FOLDER_WRITE = 'res_myAlign'

if not exists(FOLDER_WRITE):
    os.mkdir(FOLDER_WRITE)

print("start running my alignment")
now_time = datetime.datetime.now
start_time = now_time()

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

ans = []

for embed_nondist in L_nondist:
    for embed_dist in L_dist:
        fname_nondist = path_join(r"data/Nondist/", embed_nondist)
        fname_dist = path_join(r"data/Dist/", embed_dist)
        print("% s-by-% s started" % (embed_dist, embed_nondist), (now_time() - start_time).seconds)
        # cos_sim = align_two_file(fname_nondist, fname_dist, FNAME_WORDLIST, FOLDER_WRITE)
        cos_sim = align_two_file(fname_dist, fname_nondist, FNAME_WORDLIST, FOLDER_WRITE)
        ans.append([embed_nondist.split(".txt")[0], embed_dist.split(".txt")[0], cos_sim])
        print("% s-by-% s finished" % (embed_dist, embed_nondist), (now_time() - start_time).seconds)
        print(ans[-1])

for embed_nondist in L_nondist_conceptored:
    for embed_dist in L_dist:
        fname_nondist = path_join(r"data/NondistConceptored/", embed_nondist)
        fname_dist = path_join(r"data/Dist/", embed_dist)
        print("% s-by-% s started" % (embed_dist, embed_nondist), (now_time() - start_time).seconds)
        # cos_sim = align_two_file(fname_nondist, fname_dist, FNAME_WORDLIST, FOLDER_WRITE)
        cos_sim = align_two_file(fname_dist, fname_nondist, FNAME_WORDLIST, FOLDER_WRITE)
        ans.append([embed_nondist.split(".txt")[0], embed_dist.split(".txt")[0], cos_sim])
        print("% s-by-% s finished" % (embed_dist, embed_nondist), (now_time() - start_time).seconds)
        print(ans[-1])

for embed_nondist in L_nondist:
    for embed_dist in L_dist_conceptored:
        fname_nondist = path_join(r"data/Nondist/", embed_nondist)
        fname_dist = path_join(r"data/DistConceptored/", embed_dist)
        print("% s-by-% s started" % (embed_dist, embed_nondist), (now_time() - start_time).seconds)
        # cos_sim = align_two_file(fname_nondist, fname_dist, FNAME_WORDLIST, FOLDER_WRITE)
        cos_sim = align_two_file(fname_dist, fname_nondist, FNAME_WORDLIST, FOLDER_WRITE)
        ans.append([embed_nondist.split(".txt")[0], embed_dist.split(".txt")[0], cos_sim])
        print("% s-by-% s finished" % (embed_dist, embed_nondist), (now_time() - start_time).seconds)
        print(ans[-1])

for embed_nondist in L_nondist_conceptored:
    for embed_dist in L_dist_conceptored:
        fname_nondist = path_join(r"data/NondistConceptored/", embed_nondist)
        fname_dist = path_join(r"data/DistConceptored/", embed_dist)
        print("% s-by-% s started" % (embed_dist, embed_nondist), (now_time() - start_time).seconds)
        # cos_sim = align_two_file(fname_nondist, fname_dist, FNAME_WORDLIST, FOLDER_WRITE)
        cos_sim = align_two_file(fname_dist, fname_nondist, FNAME_WORDLIST, FOLDER_WRITE)
        ans.append([embed_nondist.split(".txt")[0], embed_dist.split(".txt")[0], cos_sim])
        print("% s-by-% s finished" % (embed_dist, embed_nondist), (now_time() - start_time).seconds)
        print(ans[-1])

with open("res_myAlignment_cosSim.txt", "w") as fw:
    for line in ans:
        fw.write("\t".join(map(str, line)) + "\n")

print("running finished, running time =", (now_time() - start_time).seconds)


