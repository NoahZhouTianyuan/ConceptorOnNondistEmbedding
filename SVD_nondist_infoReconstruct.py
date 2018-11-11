

import os
from os.path import join as path_join, exists
import gzip
import datetime


import numpy as np
from numpy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix, csr_matrix

now_time = datetime.datetime.now
start_time = now_time()

IS_WIKI200 = False

wiki_tail = "_wiki200" if IS_WIKI200 else ""

folder_write = "res_svdInfoReconstruct"
if not exists(folder_write):
    os.mkdir(folder_write)

wordlist = set()
if IS_WIKI200:
    with open("enwiki_vocab_min200.txt") as ff:
        for line in ff:
            wordlist.add(line.strip().split(" ")[0])

folder = "lexicons"
D_fname_feat = dict()
D_feat_fname = dict()
vectors = {}
all_feat = {}
L_fname_lex = os.listdir(folder)
for fname in L_fname_lex:
    #print(fname)
    D_fname_feat[fname.split(".txt")[0]] = set()
    with open(path_join(folder, fname)) as f_lex:
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
                    feat = feat+'.'+pos
                vectors[word][feat] = 1
                if feat not in all_feat:
                    all_feat[feat] = len(all_feat)
                D_fname_feat[fname.split(".txt")[0]].add(feat)
                D_feat_fname[feat] = fname.split(".txt")[0]

L_col_feat = []
with open(path_join(folder_write, "col_feat_fname.txt"), "w") as fw:
    for tmp_feat in all_feat:
        fw.write("\t".join([str(all_feat[tmp_feat]), tmp_feat, D_feat_fname[tmp_feat]]) + "\n")
        L_col_feat.append([all_feat[tmp_feat], tmp_feat, D_feat_fname[tmp_feat]])
L_col_feat.sort(key = lambda x: x[0])

L_word, data, row_ind, col_ind = [], [], [], []
for cc, word in enumerate(vectors):
    #if cc > 5000: break
    if IS_WIKI200 and word not in wordlist:
        continue
    L_word.append(word)
    for feat in vectors[word]:
        data.append(1)
        col_ind.append(all_feat[feat])
        row_ind.append(len(L_word) - 1)
    #if cc % 1000 == 0:
    #    print(cc, (now_time() - start_time).seconds)
#print(cc, (now_time() - start_time).seconds)

ncol = len(all_feat)
nrow = len(L_word)
A = csc_matrix((data, (row_ind, col_ind)),
           shape = (len(L_word), ncol),
           dtype = float)
del data, row_ind, col_ind
print("reading, finished, nrow = % s, ncol = % s" % (nrow, ncol))

L_k_fileR2 = []

n_top_sv = 5
u, s, vt = svds(A, k = n_top_sv)
print("svd top finished", (now_time() - start_time).seconds)
for k in range(n_top_sv):
    tmp_s = np.zeros(shape = n_top_sv)
    tmp_s[k] = s[k]
    L_col = list(range(ncol))
    L_col_chunk = []
    chunk_size = 1000
    for ii in range(int(len(L_col) / chunk_size) + 1):
        L_col_chunk.append(L_col[ii * chunk_size: (ii + 1) * chunk_size])

    L_featR2 = []
    for cc, col_chunk in enumerate(L_col_chunk):
        tmp_A = A[:, col_chunk].tocoo()
        tmp_A_reconstruct = np.array(u @ np.diag(tmp_s) @ vt[:, col_chunk])
        #print(tmp_A_reconstruct.shape, u.shape, np.diag(tmp_s).shape, vt[:, col_chunk].shape)
        tmp_L_A_colsum = [0] * len(col_chunk)
        for i, j, v in zip(tmp_A.row, tmp_A.col, tmp_A.data):
            tmp_A_reconstruct[i, j] -= v
            tmp_L_A_colsum[j] += 1
        tmp_rss = (tmp_A_reconstruct ** 2).mean(axis=0)
        tmp_tss = np.array([(i / nrow) * (1 - i / nrow) for i in tmp_L_A_colsum])
        tmp_L_r2 = 1 - tmp_rss / tmp_tss
        # print(tmp.shape)
        L_featR2.extend(tmp_L_r2)
        if cc % 10 == 0 or cc == len(L_col_chunk) - 1:
            print(cc, len(L_featR2), (now_time() - start_time).seconds)

    with open(path_join(folder_write, "feat_r2_top% s% s.txt" % (k + 1, wiki_tail)), "w") as fw:
        for tmp_feat_line, tmp_r2 in zip(L_col_feat, L_featR2):
            fw.write("\t".join(map(str, tmp_feat_line)) + "\t" + str(round(tmp_r2, 5)) + "\n")

    D_fname_featR2 = dict()
    tmp_k_fileR2 = ["top% s(% s)" % (k, round(s[k], 2))]
    for tmp_feat_line, tmp_r2 in zip(L_col_feat, L_featR2):
        tmp_fname = tmp_feat_line[2]
        if tmp_fname not in D_fname_featR2:
            D_fname_featR2[tmp_fname] = []
        if not np.isnan(tmp_r2):
            D_fname_featR2[tmp_fname].append(tmp_r2)
    for tmp_fname in L_fname_lex:
        print(tmp_fname, round(np.mean(D_fname_featR2[tmp_fname.split(".txt")[0]]), 5))
        tmp_k_fileR2.append(round(np.mean(D_fname_featR2[tmp_fname.split(".txt")[0]]), 5))
    L_k_fileR2.append(tmp_k_fileR2)
    print("topk = % s finished" % (k + 1), (now_time() - start_time).seconds)


for k in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    print("start running k = % s" % k)
    u, s, vt = svds(A, k = k)
    print("svd finished", (now_time() - start_time).seconds)

    L_col = list(range(ncol))
    L_col_chunk = []
    chunk_size = 1000
    for ii in range(int(len(L_col) / chunk_size) + 1):
        L_col_chunk.append(L_col[ii * chunk_size: (ii + 1) * chunk_size])

    L_featR2 = []
    for cc, col_chunk in enumerate(L_col_chunk):
        tmp_A = A[:, col_chunk].tocoo()
        tmp_A_reconstruct = np.array(u @ np.diag(s) @ vt[:, col_chunk])
        tmp_L_A_colsum = [0] * len(col_chunk)
        for i, j, v in zip(tmp_A.row, tmp_A.col, tmp_A.data):
            tmp_A_reconstruct[i, j] -= v
            tmp_L_A_colsum[j] += 1
        tmp_rss = (tmp_A_reconstruct ** 2).mean(axis = 0)
        tmp_tss = np.array([(i / nrow) * (1 - i / nrow) for i in tmp_L_A_colsum])
        tmp_L_r2 = 1 - tmp_rss / tmp_tss
        # print(tmp.shape)
        L_featR2.extend(tmp_L_r2)
        if cc % 10 == 0 or cc == len(L_col_chunk) - 1:
            print(cc, len(L_featR2), (now_time() - start_time).seconds)

    with open(path_join(folder_write, "feat_r2_k% s% s.txt" % (k, wiki_tail)), "w") as fw:
        for cc, (tmp_feat_line, tmp_r2) in enumerate(zip(L_col_feat, L_featR2)):
            fw.write("\t".join(map(str, tmp_feat_line)) + "\t" + str(round(tmp_r2, 5)) + "\n")

    tmp_k_fileR2 = ["% s(% s)" % (k, round(np.mean(s), 2))]
    D_fname_featR2 = dict()
    for tmp_feat_line, tmp_r2 in zip(L_col_feat, L_featR2):
        tmp_fname = tmp_feat_line[2]
        if tmp_fname not in D_fname_featR2:
            D_fname_featR2[tmp_fname] = []
        if not np.isnan(tmp_r2):
            D_fname_featR2[tmp_fname].append(tmp_r2)
    for tmp_fname in L_fname_lex:
        print(tmp_fname, round(np.mean(D_fname_featR2[tmp_fname.split(".txt")[0]]), 5))
        tmp_k_fileR2.append(round(np.mean(D_fname_featR2[tmp_fname.split(".txt")[0]]), 5))
    L_k_fileR2.append(tmp_k_fileR2)

    print("k = % s finished" % k, (now_time() - start_time).seconds)

with open(path_join(folder_write, "SVDcomp_byLexFile% s.txt" % wiki_tail), "w") as fw:
    fw.write("k" + "\t" + "\t".join([i.split(".txt")[0] for i in L_fname_lex]) + "\n")
    for line in L_k_fileR2:
        fw.write("\t".join(map(str, line)) + "\n")

print("running finished, running time =", (now_time() - start_time).seconds)
