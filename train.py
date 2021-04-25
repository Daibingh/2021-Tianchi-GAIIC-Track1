import torch.nn as nn
from utils.utils import *
import torch
import sys
import numpy as np 
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from torch.nn import BCELoss
import json
import argparse
from os.path import join as J
import os
from model.baseline import *
from model.bert.bertmodel import *
from model.nezha.nezhamodel import *
from model.nezha.modeling_nezha import *
from utils.trainer import Trainer
from dataset.dataset import *
from dataset.bertdataset import *
from utils.misc import *
import time
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
from utils.lookahead import Lookahead
import copy


def crt_model(F):

    name = F.model_name.lower()
    if name == "gru":
        model = GRU_model(
            emb_size=F.emb_size,
            wv_model_file=F.wv_model_file,
            h_size=F.rnn_dim,
            num_label=F.num_label,
            fix_emb=F.fix_emb,
            dropout=F.dropout,
            use_pretrain=F.emb_pretrain,
            bidirectional=F.rnn_bid,
            num_layer=F.rnn_num_layer
        )
    elif name == "rcnn":
        model = RCNN(
            emb_size=F.emb_size,
            wv_model_file=F.wv_model_file,
            h_size=F.rnn_dim,
            num_label=F.num_label,
            fix_emb=F.fix_emb,
            dropout=F.dropout,
            use_pretrain=F.emb_pretrain,
            num_layer=F.rnn_num_layer,
        )
    elif name == "bigru_atten2":
        model = BiGRU_Atten2(
            emb_size=F.emb_size,
            wv_model_file=F.wv_model_file,
            h_size=F.rnn_dim,
            num_label=F.num_label,
            fix_emb=F.fix_emb,
            dropout=F.dropout,
            use_pretrain=F.emb_pretrain,
            num_layer=F.rnn_num_layer,
        )
    elif name == "seq2seqatten":
        model = Seq2SeqAtten(
            emb_size=F.emb_size,
            wv_model_file=F.wv_model_file,
            h_size=F.rnn_dim,
            num_label=F.num_label,
            fix_emb=F.fix_emb,
            use_pretrain=F.emb_pretrain,
            dropout=F.dropout,
            num_layer=F.rnn_num_layer,
            bidirectional=F.rnn_bid,
            )
    elif name == "seq2seqatten2":
        model = Seq2SeqAtten2(
            emb_size=F.emb_size,
            wv_model_file=F.wv_model_file,
            h_size=F.rnn_dim,
            num_label=F.num_label,
            fix_emb=F.fix_emb,
            use_pretrain=F.emb_pretrain,
            dropout=F.dropout,
            num_layer=F.rnn_num_layer,
            bidirectional=F.rnn_bid,
            )
    elif name == "maskgru":
        model = MaskGRU_model(
            emb_size=F.emb_size,
            wv_model_file=F.wv_model_file,
            h_size=F.rnn_dim,
            num_label=F.num_label,
            fix_emb=F.fix_emb,
            use_pretrain=F.emb_pretrain,
            dropout=F.dropout,
            bidirectional=F.rnn_bid,
            )
    elif name == "bert":
        vocab = read_pkl(F.vocab_file)
        model = BERT_model(
            vocab_size=len(vocab),
            h_size=F.h_size,
            n_layer=F.n_layer,
            n_head=F.n_head,
            num_label=F.num_label,
            dropout=F.dropout,
        )
    elif name == "nezha":
        vocab = read_pkl(F.vocab_file)
        F.vocab_size = len(vocab)
        conf = BertConfig.from_dict(F.__dict__)
        model = BertForSequenceClassification(conf, F.num_label)
    else:
        print("model not found!")
        sys.exit(-1)
    
    return model


if __name__ == "__main__":

    T = Trainer()
    parser = T.get_parser()

    F = parser.parse_args()

    if F.config_file is not None:
        load_config(F.config_file, F)

    # show_config(F)
    # sys.exit(0)
        
    setup_seed(F.random_seed)

    if F.model_name in ['nezha', 'bert']:
        vocab = read_pkl(F.vocab_file)
        token_range = [5, len(vocab)]
        dataset = BERTDataset(
                            vocab,
                            data_file=F.data_file,
                            label_file=F.label_file,
                            pretrain=False,
                        )
    else:
        token_range = [1, 859]
        dataset = MyDataset(
                F.data_file,
                F.label_file,
                F.wv_model_file,
                reverse=F.seq_rever,
                # rep_prob=F.seq_rep_prob,
        )
    
    if F.debug:
        dataset = dataset.subset(dataset.index[:100])
        F.batch_size = 10
        F.enable_logging = False
        F.enable_saving = False
        F.epochs = 2
        # F.device = "cpu"
        # F.workers = 0
        F.n_fold = -1
        F.verbose = 1

    device = torch.device(F.device)

    forward_batch_fun = None
    train_step_fun = None
    if F.model_name.lower() == "nezha":
        forward_batch_fun = nezha_forward
        if F.use_fgm:
            train_step_fun = nezha_train_step_with_fgm

    if F.n_fold == -1:
        dataset_tr, dataset_val = dataset.split(F.dataset_splits, shuffle=True)

        # dataset_val.rep_prob = 0

        dl_tr = DataLoader( dataset_tr,
            batch_size=F.batch_size,
            drop_last=True,
            shuffle=F.shuffle_dataset,
            num_workers=F.workers,
            collate_fn=lambda b: collect_fn(b, 
                                            desc_max_len=F.desc_max_len, 
                                            seq_pad_meth=F.seq_pad_meth, 
                                            seq_mask_ratio=F.seq_mask_ratio, 
                                            seq_rep_prob=F.seq_rep_prob,
                                            token_range=token_range,
                                            ),
        )

        if len(dataset_val) == 0:
            dataset_val = copy.deepcopy(dataset.subset(dataset.index[:1000]))
        dl_val = DataLoader( dataset_val,
            batch_size=F.batch_size*4,
            drop_last=False,
            shuffle=F.shuffle_dataset,
            num_workers=F.workers,
            collate_fn=lambda b: collect_fn(b,
                                            desc_max_len=F.desc_max_len, 
                                            seq_pad_meth=F.seq_pad_meth, 
                                            seq_mask_ratio=0, 
                                            seq_rep_prob=0,
                                            token_range=token_range,
                                            ),
        )

        model = crt_model(F).to(device)

        if F.pretrain_model_file is not None:
            model.load_state_dict( torch.load(F.pretrain_model_file), strict=False )

        # base_opt = torch.optim.AdamW(lr=F.lr, params=model.parameters(), weight_decay=F.weight_decay)
        # lookahead = Lookahead(base_opt, k=5, alpha=0.5)
        # lr_scheduler = LambdaLR(base_opt, lr_lambda=lambda epoch: warmup_only(epoch))
        # lr_scheduler = CosineAnnealingWarmRestarts(base_opt, T_0=F.T_0, T_mult=1)

        T.train(F, 
                model, 
                dl_tr, 
                dl_val, 
                forward_batch_fun=forward_batch_fun, 
                hold_best_model=False,
                stop_cond=lambda sc: sc['val_score'] > F.val_score_limit ,
                # optimizer=base_opt,
                # lr_scheduler=lr_scheduler,
                step_fun=train_step_fun,
                verbose=F.verbose,
                )

    else:

        score_list = []

        folder_id = F.folder_id
        for i, (dataset_tr, dataset_val) in enumerate(dataset.n_fold_split(F.n_fold, shuffle=F.shuffle_dataset)):

            # dataset_val.rep_prob = 0

            if i+1 < F.fold_start: continue

            dl_tr = DataLoader(
                dataset_tr,
                batch_size=F.batch_size,
                drop_last=True,
                shuffle=F.shuffle_dataset,
                num_workers=F.workers,
                collate_fn=lambda b: collect_fn(b, 
                                                desc_max_len=F.desc_max_len, 
                                                seq_pad_meth=F.seq_pad_meth, 
                                                seq_mask_ratio=F.seq_mask_ratio, 
                                                seq_rep_prob=F.seq_rep_prob,
                                                token_range=token_range,
                                                ),
            )
            dl_val = DataLoader(
                dataset_val,
                batch_size=F.batch_size*4,
                drop_last=False,
                shuffle=F.shuffle_dataset,
                num_workers=F.workers,
                collate_fn=lambda b: collect_fn(b, 
                                                desc_max_len=F.desc_max_len, 
                                                seq_pad_meth=F.seq_pad_meth, 
                                                seq_mask_ratio=0, 
                                                seq_rep_prob=0,
                                                token_range=token_range,
                                                ),
            )
            
            model = crt_model(F).to(device)

            if F.pretrain_model_file is not None:
                model.load_state_dict( torch.load(F.pretrain_model_file), strict=False )

            F.folder_id = "{}_nfold{}-{}".format(folder_id, F.n_fold, i+1)

            base_opt = torch.optim.AdamW(lr=F.lr, params=model.parameters(), weight_decay=F.weight_decay)
            # lookahead = Lookahead(base_opt, k=5, alpha=0.5)
            lr_scheduler = LambdaLR(base_opt, lr_lambda=lambda epoch:  cosine_lr(epoch, max_epoch=F.epochs) )
            # lr_scheduler = CosineAnnealingWarmRestarts(base_opt, T_0=F.T_0, T_mult=1)

            T.train(F, 
                    model, 
                    dl_tr, 
                    dl_val, 
                    forward_batch_fun=forward_batch_fun, 
                    hold_best_model=False,
                    stop_cond=lambda sc: sc['val_score'] > F.val_score_limit ,
                    optimizer=base_opt,
                    lr_scheduler=lr_scheduler,
                    step_fun=train_step_fun,
                    verbose=F.verbose,
                    )

            del model
            torch.cuda.empty_cache()

            score_list.append( T.best_score )
            score_list[-1]['epoch'] = T.best_epoch
        
        mean_score = { k : np.mean( [ t[k] for t in score_list ] ) for k in score_list[0] }
        print(mean_score)
        if F.enable_saving:
            with open(J(F.saving_path, "mean_score.json"), 'w', encoding='utf-8') as f:
                json.dump(mean_score, f, indent=4)