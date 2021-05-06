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
from transformers import BertConfig, BertForMaskedLM
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
        load_config(F.config_file, F, 
                    ignore_keys= [
                        'debug'
                    ]
        )

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


    # dataset = dataset.subset(dataset.index[:100])
    # F.enable_logging = False
    # F.enable_saving = False
    # F.batch_size = 2
    
    if F.debug:
        dataset = dataset.subset(dataset.index[:10])
        F.enable_logging = False
        # F.enable_saving = False
        F.batch_size = 2
        F.epochs = 1
        F.device = 'cpu'
        # F.workers = 0
        F.verbose = 1

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
    
    F.vocab_size = len(vocab)
    conf = BertConfig(**F.__dict__)
    model = BertForMaskedLM(conf).to(device)

    if F.pretrain_model_file is not None:
        model.load_state_dict(torch.load(F.pretrain_model_file))
    
    T.train(
        F,
        model,
        dl_tr,
        dl_val,
        get_loss_fun=bert_pretrain_loss,
        eval_fun=bert_pretrain_eval,
        verbose=F.verbose,
        stop_cond = lambda sc : sc['val_loss'] < 1.99,
        )