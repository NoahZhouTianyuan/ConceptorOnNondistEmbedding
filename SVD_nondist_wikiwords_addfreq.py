

import os
from os.path import join as path_join, exists
import gzip
import datetime


import numpy as np
from numpy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix


folder_write = "res"
if not exists(folder_write):
    os.mkdir(folder_write)

now_time = datetime.datetime.now
start_time = now_time()

IS_WIKI200 = True
IS_ADDFREQ = True

wordlist = set()
with open("enwiki_vocab_min200.txt") as ff:
    for line in ff:
        wordlist.add(line.strip().split(" ")[0])

D_word_freq = dict()
if IS_ADDFREQ:
    with open("enwiki_vocab_min200.txt") as ff:
        for line in ff:
            line = line.strip().split(" ")
            D_word_freq[line[0]] = int(line[1])

    max_freq, min_freq = max(D_word_freq.values()), min(D_word_freq.values())
    upper, lower = int(round(np.log(max_freq), 0)), int(round(np.log(min_freq), 0))
    head = ["freqln% s" % i for i in range(upper + 1)]

    D_word_freqlabel = dict()
    for tmp_word in D_word_freq:
        D_word_freqlabel[tmp_word] = int(round(np.log(D_word_freq[tmp_word])))
        D_word_freq[tmp_word] = [0 for i in range(upper + 1)]
        D_word_freq[tmp_word][D_word_freqlabel[tmp_word]] = 1
        D_word_freq[tmp_word] = D_word_freq[tmp_word][lower:]

    with open("my_freqlabel.txt", "w") as fw:
        for tmp_word in D_word_freqlabel:
            fw.write(tmp_word + "\t" + head[D_word_freqlabel[tmp_word]] + "\n")
            
else:
    D_word_freq = {i: [] for i in wordlist}

L_word, data, row_ind, col_ind = [], [], [], []
with gzip.open("binary-vectors.txt.gz") as ff:
    for cc, line in enumerate(ff):
        #if cc > 5000: break
        if cc % 1000 == 0:
            print(cc, (now_time() - start_time).seconds)
        line = line.decode("gbk").strip().split(" ")
        if line[0] not in wordlist:
            continue
        L_word.append(line[0])
        for col, ii in enumerate(line[1:]):
            if float(ii) == 1:
                row_ind.append(len(L_word) - 1)
                col_ind.append(col)
                data.append(1)
        if IS_ADDFREQ:
            for col_freq, ii in enumerate(D_word_freq[line[0]]):
                if ii >= 1:
                    row_ind.append(len(L_word) - 1)
                    col_ind.append(col + 1 + col_freq)
                    data.append(ii)
    ncol = col + 1 + col_freq + 1 if IS_ADDFREQ else len(line) - 1

print(cc, (now_time() - start_time).seconds)
print("nrow = % s, ncol = % s" % (len(L_word), ncol))

A = csc_matrix((data, (row_ind, col_ind)),
           shape = (len(L_word), ncol),
           dtype = float)
del data, row_ind, col_ind

for k in [300]:
    u, s, vt = svds(A, k = k)
    print("svd finished", (now_time() - start_time).seconds)

    wiki_tail = "_wiki200freq" if IS_ADDFREQ else ""

    with open(path_join(folder_write, "Nondist_% sD_noSV% s.txt" % (k, wiki_tail)), "w") as ff:
        ff.write("% s % s\n" % u.shape)
        for ii, line in zip(L_word, u):
            ff.write(ii + " " + " ".join(map(str, line)) + "\n")
    print("written", (now_time() - start_time).seconds)

    with open(path_join(folder_write, "Nondist_% sD_fullSV% s.txt" % (k, wiki_tail)), "w") as ff:
        ff.write("% s % s\n" % u.shape)
        for ii, line in zip(L_word, np.matmul(u, np.diag(s))):
            ff.write(ii + " " + " ".join(map(str, line)) + "\n")
    print("written", (now_time() - start_time).seconds)

    with open(path_join(folder_write, "Nondist_% sD_halfSV% s.txt" % (k, wiki_tail)), "w") as ff:
        ff.write("% s % s\n" % u.shape)
        for ii, line in zip(L_word, np.matmul(u, np.diag(np.sqrt(s)))):
            ff.write(ii + " " + " ".join(map(str, line)) + "\n")
    print("written", (now_time() - start_time).seconds)

    print("running finished, running time = ", (now_time() - start_time).seconds)
