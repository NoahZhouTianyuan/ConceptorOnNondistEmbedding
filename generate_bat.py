

L_nondist = ["Nondist_300D_fullSV.txt",
             "Nondist_300D_halfSV.txt",
             "Nondist_300D_noSV.txt",
             "Nondist_300D_fullSV_wiki200.txt",
             "Nondist_300D_halfSV_wiki200.txt",
             "Nondist_300D_noSV_wiki200.txt"]
L_dist = ["crawl-300d-2M.vec.txt",
          "glove.840B.300d.txt",
          "word2vec.txt"]

ans = []
for ii in L_nondist:
    for jj in L_dist:
        tmp = "python unsupervised.py --src_lang en --tgt_lang en --src_emb data/Nondist/% s --tgt_emb data/% s --n_refinement 5 --export txt" % (ii, jj)
        ans.append(tmp)

with open("nondist_dist_align.bat", "w") as fw:
    for line in ans:
        fw.write(line + "\n")
