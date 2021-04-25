import torch.nn as nn
from utils.utils import *
import torch
import sys
import numpy as np 
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
import json
import argparse
from os.path import join as J
import os
from utils.trainer import Trainer
from dataset.dataset import collect_fn
from dataset.bertdataset import BERTDataset
from dataset.vocab import WordVocab
from model.bert.bertmodel import *
from utils.misc import *
import time
import copy


if __name__ == '__main__':
    T = Trainer()
    parser = T.get_parser()

    F = parser.parse_args()

    # save_config(F, "bert_pretrain_config.json")

    # sys.exit(0)

    if F.config_file is not None:
        load_config(F.config_file, F)

    setup_seed(F.random_seed)

    vocab = read_pkl(F.vocab_file)

    dataset = BERTDataset(
                            vocab,
                            data_file=F.data_file,
                            test_file=F.test_file,
                            pretrain=True,
                            mask_prob=F.mask_prob,
                            mask_ngram=F.mask_ngram,
                        )

    if F.debug:
        dataset = dataset.subset(dataset.index[:100])
        F.enable_logging = False
        F.enable_saving = False
        F.batch_size = 5
        # F.workers = 0

    device = torch.device(F.device)

    dataset_tr, dataset_val = dataset.split(F.dataset_splits)

    dl_tr = DataLoader(
                        dataset_tr, 
                        batch_size=F.batch_size, 
                        num_workers=F.workers, 
                        shuffle=F.shuffle_dataset,
                        collate_fn=lambda b: collect_fn(b, desc_max_len=F.desc_max_len, seq_pad_meth='post'),
                        drop_last=True,
                    )

    if len(dataset_val) == 0:
        dataset_val = copy.deepcopy(dataset_tr.subset(dataset_tr.index[:1000]))

    dl_val = DataLoader(
                        dataset_val, 
                        batch_size=F.batch_size*4, 
                        num_workers=F.workers, 
                        shuffle=F.shuffle_dataset,
                        collate_fn=lambda b: collect_fn(b, desc_max_len=F.desc_max_len, seq_pad_meth='post'),
                        drop_last=False,
                    )
    
    model = BERT_model(
                    vocab_size=len(vocab),
                    h_size=F.h_size,
                    n_layer=F.n_layer,
                    n_head=F.n_head,
                    dropout=F.dropout,
                    ).to(device)

    if F.init_model_file is not None:
        model.load_state_dict(torch.load(F.init_model_file))
    
    T.train(
        F,
        model,
        dl_tr,
        dl_val,
        forward_batch_fun=bert_pretrain_forward,
        get_loss_fun=bert_pretrain_loss,
        eval_fun=bert_pretrain_eval,
        verbose=F.verbose,
        )