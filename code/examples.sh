### visualization tsne and clustering

## visualization using our best model
python3 w2v_load.py -m ../models/model_native63_cbow.gensim -f gensim -a spectral -e 4 -o test
## visualization using Pohl et al. model
# python3 w2v_load.py -m ../models/model_pohl_full.w2v -f w2v -a spectral -e 0.3 -o test

