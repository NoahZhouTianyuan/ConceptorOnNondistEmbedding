

import gzip
import datetime


import numpy as np
from numpy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix


now_time = datetime.datetime.now
start_time = now_time()

L_word, data, row_ind, col_ind = [], [], [], []
with gzip.open("binary-vectors.txt.gz") as ff:
    for cc, line in enumerate(ff):
        line = line.decode("gbk").strip().split(" ")
        L_word.append(line[0])
        for col, ii in enumerate(line[1:]):
            if float(ii) == 1:
                row_ind.append(cc)
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
u, s, vt = svds(A, k = 300)
print("svd finished", (now_time() - start_time).seconds)

with open("Nondist300D_noSV.txt", "w") as ff:
    for ii, line in zip(L_word, u):
        ff.write(ii + " " + " ".join(map(str, line)) + "\n")
print("written", (now_time() - start_time).seconds)

with open("Nondist300D_fullSV.txt", "w") as ff:
    for ii, line in zip(L_word, np.matmul(u, np.diag(s))):
        ff.write(ii + " " + " ".join(map(str, line)) + "\n")
print("written", (now_time() - start_time).seconds)

with open("Nondist300D_halfSV.txt", "w") as ff:
    for ii, line in zip(L_word, np.matmul(u, np.diag(np.sqrt(s)))):
        ff.write(ii + " " + " ".join(map(str, line)) + "\n")
print("written", (now_time() - start_time).seconds)

print("running finished, running time = ", (now_time() - start_time).seconds)
        
