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
epochs = 5


if __name__ == '__main__':

    train_df = pd.read_csv( J( data_path, "rd1_train.csv" ), index_col=0 )
    test_df = pd.read_csv( J( data_path, "rd1_testA.csv" ), index_col=0 )

    train_rpt_list = [ [ c for c in t.strip().split() ] for t in train_df.desc.tolist() ]
    test_rpt_list =  [ [ c for c in t.strip().split() ] for t in test_df.desc.tolist() ]
    rpt_list = train_rpt_list + test_rpt_list
    print( "number of rpt:", len( rpt_list ) )

    # logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    # logging.root.setLevel(level=logging.INFO)

    wv_model = Word2Vec( size=embedding_size, min_count=1, sg=1, workers=4, window=window )
    wv_model.build_vocab( rpt_list )
    # wv_model.train( rpt_list, total_examples=wv_model.corpus_count, epochs=epochs )
    to_pkl( wv_model, J( data_path, "wv_model_128.pkl") )
    print("num words:", len(wv_model.wv.index2word))
