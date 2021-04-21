import torch
import numpy as np
from utils.utils import *
import pandas as pd 
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
#     seq_rep_prob = kws['seq_rep_prob']
#     for k in batch[0].keys():
#         if k == 'desc':
#             padded = torch.tensor( pad_sequences([ b[k] for b in batch ], maxlen=desc_max_len, padding=seq_pad_meth, truncating=seq_pad_meth), dtype=torch.long )
#             # padded[torch.rand(padded.shape[0], padded.shape[1]) < seq_mask_ratio] = 0
#             if seq_rep_prob > 0:
#                 mask = padded > 0
#                 p = torch.zeros(padded.shape)
#                 p[mask] = seq_rep_prob
#                 ind = torch.bernoulli(p).bool()
#                 p = torch.zeros(padded.shape)
#                 p[ind] = seq_mask_ratio
#                 ind2 = torch.bernoulli(p).bool()
#                 padded[ind2] = 0
#                 ind3 = ind & ~ind2
#                 padded[ind3] = torch.randint(1, 859, (ind3.sum(),))
#             batch_mer[k] = padded
#         else:
#             batch_mer[k] = torch.tensor( [b[k] for b in batch], dtype=torch.float32 )
#     return batch_mer


def collect_fn(batch, **kws ):
    assert type(batch[0]) == dict
    batch_mer = {}
    desc_max_len = kws.get('desc_max_len')
    seq_pad_meth = kws.get('seq_pad_meth', 'post')
    seq_mask_ratio = kws.get('seq_mask_ratio', 0)
    seq_rep_prob = kws.get('seq_rep_prob', 0)
    token_range = kws.get("token_range", [0,1])
    for k in batch[0].keys():
        if k in ['desc', 'bert_input']:
            padded = torch.tensor( pad_sequences([ b[k] for b in batch ], maxlen=desc_max_len, padding=seq_pad_meth, truncating=seq_pad_meth), dtype=torch.long )
            # padded[torch.rand(padded.shape[0], padded.shape[1]) < seq_mask_ratio] = 0
            if k == 'desc' and seq_rep_prob > 0:
                mask = padded > 0
                p = torch.zeros(padded.shape)
                p[mask] = seq_rep_prob
                ind = torch.bernoulli(p).bool()
                p = torch.zeros(padded.shape)
                p[ind] = seq_mask_ratio
                ind2 = torch.bernoulli(p).bool()
                padded[ind2] = 0
                ind3 = ind & ~ind2
                padded[ind3] = torch.randint(token_range[0], token_range[1], (ind3.sum(),))
                batch_mer[k] = padded
            else:
                batch_mer[k] = padded
        elif k == 'bert_label':
            batch_mer[k] = torch.tensor( pad_sequences( [b[k] for b in batch], maxlen=desc_max_len, padding=seq_pad_meth, truncating=seq_pad_meth ), dtype=torch.long )
        else:
            batch_mer[k] = torch.tensor( [b[k] for b in batch], dtype=torch.float32 )
    return batch_mer


class MyDataset(Dataset):

    def __init__(self, 
                data_file, 
                label_file,
                wv_model_file,
                reverse=False,
                # rep_prob=0,
                ):
        super().__init__()
        if type(data_file) == str:
            self.data = pd.read_csv(data_file, index_col=0)
        else:
            assert type(data_file) == list
            d = []
            for f in data_file:
                d.append( pd.read_csv(f, index_col=0) )
            self.data = pd.concat(d, axis=0, ignore_index=True)

        wv_model = read_pkl(wv_model_file)
        self.word2index = {w:i+1 for i, w in enumerate(wv_model.wv.index2word)}
        self.index2word = ['<pad>'] + wv_model.wv.index2word
        self.data['desc'] = self.data['desc'].map(lambda line: [ self.word2index.get(t, 0) for t in line.strip().split() ] )
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

        self.reverse = reverse
        # self.rep_prob = rep_prob
        self.wv_model = wv_model

    @property
    def index(self):
        return list(range(len(self)))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = {}
        desc = self.data.iloc[idx]['desc']
        # if self.rep_prob > 0:
            # desc = [ t if np.random.rand() > self.rep_prob else self.word2index[ self.wv_model.wv.similar_by_word( self.index2word[t], 1)[0][0] ]  for t in desc  ]
            # desc = [ t if np.random.rand() > self.rep_prob else np.random.randint( 0, len(self.index2word) )  for t in desc  ]
        if self.reverse: 
            desc = desc[::-1]
        item['desc'] = desc
        if self.label is not None:
            item['label'] = self.label.iloc[idx].tolist()
        return item

    def select(self, idx):
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
                dataset_splits.append(self.select(idx[start:]))
            else:
                dataset_splits.append(self.select(idx[start:start+num]))
            start += num
        return dataset_splits

    def n_fold_split(self, n_fold=5, shuffle=False):
        idx = self.index
        for tr_idx, val_idx in KFold(n_fold, shuffle=shuffle).split(idx):
            yield self.select(tr_idx), self.select(val_idx)


class MyDataset2(Dataset):

    def __init__(self, 
                train_file, 
                test_file,
                label_file,
                max_feat,
                is_train=True,
                min_df=1,
                max_df=1.0,
                token_pat=r"(?u)\b\w\w+\b",
                ):
        super().__init__()
        train = pd.read_csv(train_file, index_col=0)
        test = pd.read_csv(test_file, index_col=0)
        self.data = train if is_train else test
        self.label = None
        if label_file is not None:
            self.label = pd.read_csv(label_file, index_col=0)
        self.tv = TfidfVectorizer(token_pattern=token_pat, max_df=max_df, min_df=min_df, max_features=max_feat)
        self.tv.fit(train.desc.tolist()+test.desc.tolist())
        self.num_feat = len(self.tv.get_feature_names()) + 1

    def __len__(self):
        return self.data.shape[0]

    @property
    def index(self):
        return self.data.index.tolist()

    def __getitem__(self, idx):
        rid = self.data.index[idx]
        desc = self.data.loc[rid, 'desc']
        x = torch.zeros(self.num_feat, dtype=torch.float32)
        x[0] = len(desc.strip().split()) / 100.
        x[1:] = torch.tensor( self.tv.transform([desc]).toarray()[0], dtype=torch.float32)
        item = {"x": x}
        if self.label is not None:
            item['label'] = torch.tensor(self.label.loc[rid].values, dtype=torch.float32)
        return item

    def select(self, rid):
        d = copy.deepcopy(self)
        d.data = d.data.loc[rid]
        if d.label is not None:
            d.label = d.label.loc[rid]
        return d

    def split(self, splits, shuffle=False):
        rid = self.data.index.tolist()
        if shuffle:
            np.random.shuffle(rid)
        dataset_splits = []
        start = 0
        for i, sp in enumerate(splits):
            num = int(len(self) / sum(splits) * sp)
            if i == len(splits)-1:
                dataset_splits.append(self.select(rid[start:]))
            else:
                dataset_splits.append(self.select(rid[start:start+num]))
            start += num
        return dataset_splits

    def n_fold_split(self, n_fold=5, shuffle=False):
        rid = self.index
        for tr_idx, val_idx in KFold(n_fold, shuffle=shuffle).split(rid):
            yield self.select([rid[i] for i in tr_idx]), self.select([rid[i] for i in val_idx])


class MyDataset3(Dataset):

    def __init__(self, 
                train_file, 
                test_file,
                label_file,
                wv_model_file,
                max_len,
                max_feat,
                is_train=True,
                min_df=1,
                max_df=1.0,
                token_pat=r"(?u)\b\w\w+\b",
                ):
        super().__init__()
        wv_model = read_pkl(wv_model_file)
        self.word2index = {w:i+1 for i, w in enumerate(wv_model.wv.index2word)}
        train = pd.read_csv(train_file, index_col=0)
        test = pd.read_csv(test_file, index_col=0)
        self.data = train if is_train else test
        self.label = None
        if label_file is not None:
            self.label = pd.read_csv(label_file, index_col=0)
        self.tv = TfidfVectorizer(token_pattern=token_pat, max_df=max_df, min_df=min_df, max_features=max_feat)
        self.tv.fit(train.desc.tolist()+test.desc.tolist())
        self.num_feat = len(self.tv.get_feature_names()) + 1
        self.max_len = max_len

    def __len__(self):
        return self.data.shape[0]

    @property
    def index(self):
        return self.data.index.tolist()

    def __getitem__(self, idx):
        rid = self.data.index[idx]
        desc = self.data.loc[rid, 'desc']
        x2 = torch.tensor( [ self.word2index.get(t, 0) for t in desc.split()], dtype=torch.long )
        x = torch.zeros(self.max_len, dtype=torch.long)
        len_ = min(self.max_len, x2.shape[0])
        x[:len_] = x2[:len_]
        item = {'x2': x}

        x = torch.zeros(self.num_feat, dtype=torch.float32)
        x[0] = len(desc.strip().split()) / 100.
        x[1:] = torch.tensor( self.tv.transform([desc]).toarray()[0], dtype=torch.float32)
        item['x1'] = x
        if self.label is not None:
            item['label'] = torch.tensor(self.label.loc[rid].values, dtype=torch.float32)
        return item

    def select(self, rid):
        d = copy.deepcopy(self)
        d.data = d.data.loc[rid]
        if d.label is not None:
            d.label = d.label.loc[rid]
        return d

    def split(self, splits, shuffle=False):
        rid = self.data.index.tolist()
        if shuffle:
            np.random.shuffle(rid)
        dataset_splits = []
        start = 0
        for i, sp in enumerate(splits):
            num = int(len(self) / sum(splits) * sp)
            if i == len(splits)-1:
                dataset_splits.append(self.select(rid[start:]))
            else:
                dataset_splits.append(self.select(rid[start:start+num]))
            start += num
        return dataset_splits

    def n_fold_split(self, n_fold=5, shuffle=False):
        rid = self.index
        for tr_idx, val_idx in KFold(n_fold, shuffle=shuffle).split(rid):
            yield self.select([rid[i] for i in tr_idx]), self.select([rid[i] for i in val_idx])


def crt_dataset(F):
    if F.model_name.lower() == 'nn':
        d = MyDataset2(
                train_file=F.train_file, 
                test_file=F.test_file,
                label_file=F.label_file,
                max_feat=F.max_feat,
                is_train=True if F.label_file is not None else False,
                min_df=5,
                max_df=0.95,
                token_pat=r"\S+",
        )
        F.feat_dim = d.num_feat
        return d
    elif F.model_name.lower() == 'nn_gru':
        d = MyDataset3(
                wv_model_file=F.wv_model_file,
                max_len=F.desc_max_len,
                train_file=F.train_file, 
                test_file=F.test_file,
                label_file=F.label_file,
                max_feat=F.max_feat,
                is_train=True if F.label_file is not None else False,
                min_df=5,
                max_df=0.95,
                token_pat=r"\S+",
        )
        F.feat_dim = d.num_feat
        return d
    else:
        return MyDataset(
            F.data_file,
            F.label_file,
            F.wv_model_file,
            reverse=F.seq_rever,
            rep_prob=F.seq_rep_prob,
        )
