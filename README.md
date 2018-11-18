# ConceptorOnNondistEmbedding
Apply conceptor on vectors of nondistributional word embedding. And compare the results to show what happened.

### Outline

This package has several separated functions in different files:      

* Evaluation of embedding vectors by similarity tasks: Main.py 
* Using SVD to construct dense and low-dim nondistributional word vectors: SVD\_nondist\_wikiwords.py and SVD\_nondist\_wikiwords\_addfreq.py
* Finding how much information remained after SVD: SVD\_nondist\_inforReconstruct.py
* Using Hashing trick instead of SVD to construct nondistributional word vectors: Hashing\_nondist.py
* Aligning two set of embedding vectors by a linear transformation: nonortho\_alginment.py.
* Evaluation the alignment by cos-similarity: alignment\_analysis.py



#### Main.py (Evaluation of embedding vectors by similarity tasks)

This py file is an script version of the preivous [ConceptorPostPro_2.ipynb](https://colab.research.google.com/drive/192rRmmG2NQCDiz84KPbUzhOA64vxmWeg#scrollTo=HWgAGFBEjZVQ). Its workflow is as follows:

1. read an embedding vector file;
2. run the similarity tasks and get scores;
3. use conceptor and repeat 2.;
4. for k=1-5, use abtt-k and repeat 2.;
5. for different embedding vector files, repeat 1.-4.;
6. summarize all results and write them into a csv file.


#### SVD\_nondist\_wikiwords.py and SVD\_nondist\_wikiwords\_addfreq.py (Using SVD to construct dense and low-dim nondistributional word vectors)

In SVD\_nondist\_wikiwords.py, we use SVD to generate low-dim dense vectors of nondistributional word vectors. Our data source is _binary-vectors.txt.gz_, got from the github project [_non-distributional_][nondist-github].

Here in total 6 kinds of low-dim vectors can be generated. We can choose whether to filter the word in [wiki200 word list][wiki200] before using SVD, and choose how much information of singular values we kept. We have 3 options for this: fullSV, halfSV, noSV, where fullSV, is the most commonly used verseion, means keep the singular values _S_, multiply it on the left singular vectors _U_ and return _U * diag(S)_, halfSV means return _U * diag(sqrt(S))_ while noSV means only return _U_. Thus in total there are 2*3=6 versions. In the downstream tasks, we find fullSV versions are mostly useful while halfSV and noSV versions are only used for comparison. And whether to filter words before SVD has little impact.

The only difference between SVD\_nondist\_wikiwords\_addfreq.py and SVD\_nondist\_wikiwords.py is that SVD\_nondist\_wikiwords\_addfreq.py can add frequency information, which is not contained in the traditional Dyer's version, into our low-dim vectors. Here we use the 


#### SVD\_nondist\_inforReconstruct.py (Finding how much information remained after SVD)

In this file, we use the lexicon files from the github project [_non-distributional_][nondist-github] and figure out to what extent the information of each file is prevented after SVD. Here we measure the percentage of information prevented of a feature by _Rsquare_, and the percentage of information prevented of a file by the average of _Rsquare_ of all features within this file.

Besides, in this file, we can also add the frequency information in the [wiki200 word list][wiki200] and convert it into logarithmic bin form. That is to say, there exist several features _freq\_i_ where _i_=1,2,... and for each word w, only _freq\_round(log(#w))_=1 and all other _freq\_i_ is equal to 0.


#### Hashing\_nondist.py (Using Hashing trick instead of SVD to construct nondistributional word vectors) 

In this py file, we use the lexicon files from the github project [_non-distributional_][nondist-github] and form another version of low-dim word vectors by standard Hashing trick. The implementation of hashing trick is by sci-kit learn, which simply hash the name of each feature. Besides, this implementation provides an option whether to use random negative sign or nonnegative sign, in case there will be strong bias in the downstream tasks especially calculating the inner product.


#### nonortho\_alginment.py (Aligning two set of embedding vectors by a linear transformation)

In this py file, we use OLS instead of [MUSE](https://github.com/facebookresearch/MUSE) to align two sets of word vectors. The cos-similarity of these alignments will also be reported in this file.


#### alignment\_analysis.py (Evaluation the alignment by cos-similarity)

In this py file, we simply calculate the cos-similarity of two sets of word vectors, typically word vectors after aligning by MUSE, for MUSE doesn't report cos-similarity after alignment.




[nondist-github]: (https://github.com/mfaruqui/non-distributional) 
[wiki200]: (https://github.com/PrincetonML/SIF/blob/master/auxiliary_data/enwiki_vocab_min200.txt)