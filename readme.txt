# Resources

This is a first readme for the resoures but we plan to make if available on ORTOLANG (part of CLARIN) with better documentation.

## Resource 1: Embeddings models

Embedding models are in the "models" folder. The best model described in the paper is "model_native63_cbow.gensim".

## Resource 2: Clusters (and visualization)

All the visualization files are in the "visualization" folder.

You can visualize and recompute clusters and tsne visualization by going to the "code" folder.
See the examples.sh file for examples of usages. Or just use the -h option in any python script.

## Folder structure

.
├── code 
│   ├── data_util
│   │   ├── df_emoji_faces_std.tsv
│   │   ├── emojichart_v5.0.tsv
│   │   ├── emojione_32px (lot of files)
│   │   └── ready_data
│   │       └── tweets_df_echo_ssth_1M_raw.ls # our list of tweets. However it is not legal to give as is. But we will give a full list of tweets id (we plan to expand the corpus for the final paper) and the code to get them by the Twitter API.
│   ├── examples.sh
│   ├── w2v_load.py
│   └── w2v.py
├── models
│   ├── model_native63_cbow.gensim   #### the main embedding resource ####
│   ├── model_native63_cbow.gensim.syn1neg.npy
│   ├── model_native63_cbow.gensim.wv.syn0.npy
│   ├── model_native63_skipgram.gensim
│   ├── model_native63_skipgram.gensim.syn1neg.npy
│   ├── model_native63_skipgram.gensim.wv.syn0.npy
│   └── model_pohl_full.w2v
├── readme.md
└── visualization
    ├── clusters_native63_cbow.html   #### the best clusters we obtained ####
    ├── clusters_native63_skipgram.html
    ├── clusters_pohl_full.html
    ├── tw_native63_cbow_size0.1.html
    ├── tw_native63_cbow_size4.html
    ├── tw_native63_skipgram_size0.1.html
    ├── tw_native63_skipgram_size4.html
    ├── tw_pohl_full_size0.3.html
    └── tw_pohl_full_size4.html

6 directories, 2689 files
