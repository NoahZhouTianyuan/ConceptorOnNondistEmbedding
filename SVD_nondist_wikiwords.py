

import os
from os.path import join as path_join, exists
import gzip
import datetime


import numpy as np
from numpy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix


folder_write = "res3"
if not exists(folder_write):
    os.mkdir(folder_write)

now_time = datetime.datetime.now
start_time = now_time()

wordlist = set()
with open("enwiki_vocab_min200.txt") as ff:
    for line in ff:
        wordlist.add(line.strip().split(" ")[0])

L_word, data, row_ind, col_ind = [], [], [], []
with gzip.open("binary-vectors.txt.gz") as ff:
    for cc, line in enumerate(ff):
        #if cc > 500: break
        line = line.decode("gbk").strip().split(" ")
        if line[0] not in wordlist:
            continue
        L_word.append(line[0])
        for col, ii in enumerate(line[1:]):
            if float(ii) == 1:
                row_ind.append(len(L_word) - 1)
                col_ind.append(col)
                data.append(1)
        if cc % 1000 == 0:
            print(cc, (now_time() - start_time).seconds)
    ncol = len(line) + 1

print(cc, (now_time() - start_time).seconds)
print("nrow = % s, ncol = % s" % (len(L_word), ncol))

A = csc_matrix((data, (row_ind, col_ind)),
           shape = (len(L_word), ncol),
           dtype = float)
del data, row_ind, col_ind

for k in [600, 700, 800, 900, 1000]:
    u, s, vt = svds(A, k = k)
    print("svd finished", (now_time() - start_time).seconds)

    with open(path_join(folder_write, "Nondist_% sD_noSV_wiki200.txt" % k), "w") as ff:
        ff.write("% s % s\n" % u.shape)
        for ii, line in zip(L_word, u):
            ff.write(ii + " " + " ".join(map(str, line)) + "\n")
    print("written", (now_time() - start_time).seconds)

    with open(path_join(folder_write, "Nondist_% sD_fullSV_wiki200.txt" % k), "w") as ff:
        ff.write("% s % s\n" % u.shape)
        for ii, line in zip(L_word, np.matmul(u, np.diag(s))):
            ff.write(ii + " " + " ".join(map(str, line)) + "\n")
    print("written", (now_time() - start_time).seconds)

    with open(path_join(folder_write, "Nondist_% sD_halfSV_wiki200.txt" % k), "w") as ff:
        ff.write("% s % s\n" % u.shape)
        for ii, line in zip(L_word, np.matmul(u, np.diag(np.sqrt(s)))):
            ff.write(ii + " " + " ".join(map(str, line)) + "\n")
    print("written", (now_time() - start_time).seconds)

    print("running finished, running time = ", (now_time() - start_time).seconds)
