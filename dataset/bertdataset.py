import tqdm
import random
import pandas as pd 
import torch
import numpy as np
from utils.utils import *
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



# def collect_fn(batch, **kws ):
#     assert type(batch[0]) == dict
#     batch_mer = {}
#     desc_max_len = kws['desc_max_len']
#     seq_pad_meth = kws['seq_pad_meth']
#     seq_mask_ratio = kws['seq_mask_ratio']
#     for k in batch[0].keys():
#         if k in ['desc', 'bert_input']:
#             padded = torch.tensor( pad_sequences([ b[k] for b in batch ], maxlen=desc_max_len, padding=seq_pad_meth, truncating=seq_pad_meth), dtype=torch.long )
#             padded[torch.rand(padded.shape[0], padded.shape[1]) < seq_mask_ratio] = 0
#             batch_mer[k] = padded
#         elif k == 'bert_label':
#             batch_mer[k] = torch.tensor( pad_sequences( [b[k] for b in batch], maxlen=desc_max_len, padding=seq_pad_meth, truncating=seq_pad_meth ), dtype=torch.long )
#         else:
#             batch_mer[k] = torch.tensor( [b[k] for b in batch], dtype=torch.float32 )
#     return batch_mer


class BERTDataset(Dataset):
    def __init__(self, 
                    vocab, 
                    data_file,
                    label_file=None,
                    test_file=None,
                    pretrain=True,
                    mask_prob=.15,
                    mask_ngram=1,
                    ):
        self.vocab = vocab
        if type(data_file) == str:
            self.data = pd.read_csv(data_file, index_col=0)  #['desc'].tolist()
        else:
            assert type(data_file) == list
            d = []
            for f in data_file:
                d.append( pd.read_csv(f, index_col=0) )
            self.data = pd.concat(d, axis=0, ignore_index=True)
        
        if pretrain and test_file is not None:
            d = [self.data]
            if type(test_file) == str:
                # self.data += pd.read_csv(test_file, index_col=0)  # ['desc'].tolist()
                d.append( pd.read_csv(test_file, index_col=0) )
            else:
                assert type(test_file) == list
                for f in test_file:
                    d.append( pd.read_csv(f, index_col=0) )
            self.data = pd.concat(d, axis=0, ignore_index=True)

        self.label = None
        if label_file is not None:
            assert type(data_file) == type(label_file)
            if type(label_file) == str:
                self.label = pd.read_csv(label_file, index_col=0)
            else:
                d = []
                for f in label_file:
                    d.append( pd.read_csv(f, index_col=0) )
                self.label = pd.concat(d, axis=0, ignore_index=True)
                assert self.label.shape[0] == self.data.shape[0]

        self.pretrain = pretrain
        self.mask_prob = mask_prob
        self.mask_ngram = mask_ngram

    def __len__(self):
        return self.data.shape[0]

    @property
    def index(self):
        return list(range(len(self)))


    def __getitem__(self, item):

        t1 = self.data.iloc[item]['desc']
        if self.pretrain:
            
            t1_random, t1_label = self.random_word(t1)

            # [CLS] tag = SOS tag, [SEP] tag = EOS tag
            t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]

            t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
            bert_input = t1
            bert_label = t1_label

            output = {"bert_input": bert_input,
                      "bert_label": bert_label,
                      }

            return output

        else:
            d = {}
            d['desc'] = [ self.vocab.stoi.get(token, self.vocab.unk_index) for token in t1.split() ]
            if self.label is not None: 
                d['label'] = self.label.iloc[item].tolist()
            return d

    def subset(self, idx):
        d = copy.deepcopy(self)
        d.data = d.data.iloc[idx]
        if d.label is not None:
            d.label = d.label.iloc[idx]
        return d

    def split(self, splits, shuffle=False):
        idx = self.index
        if shuffle:
            np.random.shuffle(idx)
        dataset_splits = []
        start = 0
        for i, sp in enumerate(splits):
            num = int(len(self) / sum(splits) * sp)
            if i == len(splits)-1:
                dataset_splits.append(self.subset(idx[start:]))
            else:
                dataset_splits.append(self.subset(idx[start:start+num]))
            start += num
        return dataset_splits

    def n_fold_split(self, n_fold=5, shuffle=False):
        idx = self.index
        for tr_idx, val_idx in KFold(n_fold, shuffle=shuffle).split(idx):
            yield self.subset(tr_idx), self.subset(val_idx)

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        i = 0

        # for i, token in enumerate(tokens):
        while i < len(tokens):

            # token = tokens[i]

            prob = random.random()
            if prob < self.mask_prob:
                
                prob /= self.mask_prob

                # 80% randomly change token to mask token
                if prob < 0.8:
                    j = i
                    while i < len(tokens) and i-j < self.mask_ngram:
                        output_label.append(self.vocab.stoi.get(tokens[i], self.vocab.unk_index))
                        tokens[i] = self.vocab.mask_index
                        i += 1
                # 10% randomly change token to random token
                elif prob < 0.9:
                    j = i
                    while i < len(tokens) and i-j < self.mask_ngram:
                        output_label.append(self.vocab.stoi.get(tokens[i], self.vocab.unk_index))
                        tokens[i] = random.randrange(len(self.vocab))
                        i += 1

                # 10% randomly change token to current token
                else:
                    j = i
                    while i < len(tokens) and i-j < self.mask_ngram:
                        tokens[i] = self.vocab.stoi.get(tokens[i], self.vocab.unk_index)
                        output_label.append(tokens[i])
                        i += 1

                # output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))
            else:
                tokens[i] = self.vocab.stoi.get(tokens[i], self.vocab.unk_index)
                output_label.append(0)
                i += 1

        return tokens, output_label
