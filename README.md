# Word2Vec Python-based node with Tensorflow
This extension contains a Python-based node which performs Word2Vec with both skip-gram and CBOW algorithms, 
letting the user choose between hierarchical softmax and negative sampling as approaches to perform the actual fit. 
The underlying engine for the fit and for some of the pre-processing is Tensorflow.

Some of the overall ideas for this implementation come from this great blog post/README file: https://github.com/chao-ji/tf-word2vec/blob/master/README.md
