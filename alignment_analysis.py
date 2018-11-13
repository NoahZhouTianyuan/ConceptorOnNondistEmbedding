

import os
from os.path import join as path_join, exists
import datetime
from itertools import chain


import numpy as np
from numpy import array
from scipy.spatial.distance import cosine


folder_alignedVec = r"data\AlignedVec"

L_nondist, L_nondist_conceptored = [], []
for sv_type in ["fullSV", "halfSV", "noSV"]:
    for wiki200_type in ["_wiki200", ""]:
        L_nondist_conceptored.append("Nondist_300D_% s% s+conceptored.txt" % (sv_type, wiki200_type))
        L_nondist.append("Nondist_300D_% s% s.txt" % (sv_type, wiki200_type))

L_dist, L_dist_conceptored = [], []
for emb in ["fasttextCrawl", "glove840B300D", "word2vec"]:
    L_dist.append("% s.txt" % emb)
    L_dist_conceptored.append("% s+conceptored.txt" % emb)

print("start running analyzing aligned vectors")
now_time = datetime.datetime.now
start_time = now_time()

wordlist = set()
with open(r"data/Wiki_vocab_gt200/enwiki_vocab_min200.txt") as ff:
    for line in ff:
        wordlist.add(line.strip().split(" ")[0])

def evalutate_pair(emb_nondist, emb_dist):

    emb_dist = emb_dist.split(".txt")[0]
    emb_nondist = emb_nondist.split(".txt")[0]
    fname_nondist = "% s-to-% s.txt" % (emb_nondist, emb_dist)
    fname_dist = "% s-from-% s.txt" % (emb_dist, emb_nondist)
    D_embType_fname = {"nondist": fname_nondist, "dist": fname_dist}
    D_embType_embName = {"nondist": emb_nondist, "dist": emb_dist}

    if not exists(path_join(folder_alignedVec, fname_nondist)):
        print("Error: no file % s" % fname_nondist)
        return


    D_embType_vec = dict()
    for embType in D_embType_fname:
        with open(path_join(folder_alignedVec, D_embType_fname[embType]), encoding="gbk") as f_nondist:
            vocab, X = [], []
            for cc, line in enumerate(f_nondist):
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
        D_embType_vec[embType] = (vocab, X)

    vocab_nondist, X_nondist = D_embType_vec["nondist"]
    vocab_dist, X_dist = D_embType_vec["dist"]
    L_cosine_sim = []
    for word in set.intersection(set(vocab_nondist), set(vocab_dist)):
        tmp_cosine_sim = cosine(X_dist[vocab_dist.index(word), :], X_nondist[vocab_nondist.index(word), :])
        if not np.isnan(tmp_cosine_sim):
            L_cosine_sim.append(1 - tmp_cosine_sim)
        else:
            pass
            #L_cosine_sim.append(0)
    #print(len(L_cosine_sim), len(set.intersection(set(vocab_nondist), set(vocab_dist))))
    return np.mean(L_cosine_sim)

ans = []
for emb_nondist in chain(L_nondist_conceptored, L_nondist):
    for emb_dist in chain(L_dist_conceptored, L_dist):
        print(emb_nondist, emb_dist, (now_time() - start_time).seconds)
        tmp_simi = evalutate_pair(emb_nondist, emb_dist)
        emb_dist = emb_dist.split(".txt")[0]
        emb_nondist = emb_nondist.split(".txt")[0]
        tmp_line = ["% s+% s" % (emb_nondist, emb_dist),
                    emb_nondist.split("+conceptored")[0],
                    emb_dist.split("+conceptored")[0],
                    ("+conceptored" in emb_nondist),
                    ("+conceptored" in emb_dist),
                    tmp_simi]
        ans.append(tmp_line)
        print(tmp_simi)

with open("aligned_similarity.txt", "w") as fw:
    fw.write("\t".join(["id", "emb_nondist", "emb_dist", "is_nondist_conceptored", "is_dist_conceptored",
                        "mean_cosine_similarity"]) + "\n")
    for line in ans:
        fw.write("\t".join(map(str, line)) + "\n")

print("running finished, running time =", (now_time() - start_time).seconds)








