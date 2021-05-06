from utils import *
import logging
from os.path import join as J
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from gensim.models import Word2Vec


data_path = 'data'
embedding_size = 128
window = 10
epochs = 20
training = False 

files = [
    "rd1_train.csv",
    "rd1_testA.csv",
    # "rd2_train.csv",
    # "rd1_testB.csv",
]


if __name__ == '__main__':

    rpt_list = []
    for f in files:
        df = pd.read_csv(J(data_path, f), index_col=0)
        rpt_list += [ [ c for c in t.strip().split() ] for t in df.desc.tolist() ]

    print( "number of rpt:", len( rpt_list ) )

    # logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    # logging.root.setLevel(level=logging.INFO)

    wv_model = Word2Vec( size=embedding_size, min_count=1, sg=1, workers=4, window=window )
    wv_model.build_vocab( rpt_list )
    if training:
        wv_model.train( rpt_list, total_examples=wv_model.corpus_count, epochs=epochs )
    to_pkl( wv_model, J( data_path, "wv_model_128.pkl") )
    print("num words:", len(wv_model.wv.index2word))
