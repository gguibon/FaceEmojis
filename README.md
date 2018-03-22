# FaceEmojis: Emoji Embeddings and Clusters of Face Emojis

This is the source code from a paper we presented to CICLing 2018:

From emoji To Categorical Emoji Prediction, Guibon et al., 2018. CICLing, Hanoi 

FaceEmojis is a continuous bag-of-words emoji embeddings from tweets and emotion clusters of face emojis. One of its purpose is to be used in emoji recommendation systems to improve them.

This resource have been created fully automatically from raw tweets to fine-grained clusters of emojis using word2vec and spectral clustering algorithms.

The resulting embedding model and the clusters are available. The resource files contain the code to replicate the methodology, the models learnt, the visualization files, and the code to replicate the visualization process.

# Resources

This is a first readme for the resoures but we plan to make if available on ORTOLANG (part of CLARIN) with more detailed better documentation.

## Resource 1: Embeddings models

Embedding models are in the ["models" folder](https://drive.google.com/open?id=0By_QEvK1tPkQUFF2V2JFRV9ZUGc). The best model described in the paper is "model_native63_cbow.gensim".

## Resource 2: Clusters (and visualization)

All the visualization files are in the "visualization" folder.

You can visualize and recompute clusters and tsne visualization by going to the "code" folder.
See the examples.sh file for examples of usages. Or just use the -h option in any python script.

