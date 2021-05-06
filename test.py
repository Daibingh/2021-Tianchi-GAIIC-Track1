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
# from model.nezha.modeling_nezha import *
from utils.trainer import Trainer
from dataset.dataset import *
from dataset.bertdataset import *
from utils.misc import *
import time
from train import crt_model
from tqdm import tqdm
from utils.trainer import _forward_batch


def predict(F, model, dl):
    device = torch.device(F.device)
    pred = []
    model.eval()
    if F.model_name.lower() == "bert":
        forward_barch_fun = bert_forward
    elif F.model_name.lower() == "nezha":
        forward_barch_fun = nezha_forward
    else:
        forward_barch_fun = _forward_batch
    with torch.no_grad():
        for b in tqdm(dl):
            # x = b['desc'].to(device)
            # y = model(x)
            y = forward_barch_fun(F, model, b)
            pred.append(y.cpu().numpy())
    
    pred = np.concatenate(pred, axis=0)
    return pred


if __name__ == "__main__":

    T = Trainer()
    parser = T.get_parser()
    parser.add_argument("--model_list_file", required=True)
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--test_file", required=True)
    F = parser.parse_args()
    test_file = F.test_file

    md_df = pd.read_csv(F.model_list_file)  # [model_path, config_path, weight]
    print( "Total model: {}".format(md_df.shape[0]) )

    pred = 0.
    weight_sum = 0.

    for i in range(md_df.shape[0]):
        model_path = md_df.iloc[i]['model_path']
        config_path = md_df.iloc[i]['config_path']
        weight = md_df.iloc[i]['weight']
        
        load_config(config_path, F)
        F.test_file = test_file

        if F.model_name.lower() in ["bert", "nezha"]:
            vocab = read_pkl(F.vocab_file)
            dataset_test = BERTDataset(
                                vocab,
                                data_file=F.test_file,
                                pretrain=False,
                            )
        else:
            dataset_test = MyDataset(F.test_file, None, F.wv_model_file, F.seq_rever)

        dl_test = DataLoader(
            dataset_test ,
            batch_size=F.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=F.workers,
            collate_fn=lambda b: collect_fn(b, desc_max_len=F.desc_max_len, seq_pad_meth=F.seq_pad_meth),
        )

        device = torch.device(F.device)
        model = crt_model(F).to(device)

        model.load_state_dict( torch.load( model_path ) )
        print("No. {}, load {} model from {}".format(i, F.model_name, model_path))

        pred += weight * predict(F, model, dl_test)
        weight_sum += weight

    pred /= weight_sum

    pred_df = pd.DataFrame(pred, index=dataset_test.index)
    pred_df.to_csv( F.out_file )
    # cvt_fmt(pred_df, F.out_file )
