import os 
import sys
path = os.path.abspath('.')
if path not in sys.path: sys.path.append(path)
from dataset.vocab import WordVocab
from utils import *


if __name__ == "__main__":

    with open("data/corpus.txt", "r", encoding='utf-8') as f:
        vocab = WordVocab(f, min_freq=1)

    to_pkl(vocab, "data/vocab.pkl")
    print("vocab len:", len(vocab))
