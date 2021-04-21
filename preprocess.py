from utils.utils import *
from utils.misc import *
from os.path import join as J 
import pandas as pd 
from io import StringIO


data_path = "/tcdata"
out_path = "data"
rd1_train_name = "track1_round1_train_20210222.csv"
rd1_testA_name = "track1_round1_testA_20210222.csv"
rd1_testB_name = "track1_round1_testB.csv"

rd2_train_name = "train.csv"
rd2_testA_name = "testA.csv"
rd2_testB_name = "testB.csv"

NUM_LABEL = 17
NUM_LABEL2 = 12

col_names = ["desc", "label", "label2"]


def func(t, n):
    a = np.zeros(n)
    a[[int(i) for i in t.split()]] = 1.0                             
    return a


def process_data(file):
    with open(file) as f:
        t = f.read().replace('|', '')

    df = pd.read_csv(StringIO(t), index_col=0, header=None)
    n_col = df.shape[1]
    df.index.name = 'rid'
    df.columns = col_names[:n_col]
    df.fillna('', inplace=True)
    ret = {}
    if "label" in df.columns:
        lab = df.label.map(lambda t: func(t, NUM_LABEL))
        lab = pd.DataFrame(np.stack(lab.values), index=lab.index)
        ret['label'] = lab
        df.drop('label', inplace=True, axis=1)
    if "label2" in df.columns:
        lab = df.label2.map(lambda t: func(t, NUM_LABEL2))
        lab = pd.DataFrame(np.stack(lab.values), index=lab.index)
        ret['label2'] = lab
        df.drop('label2', inplace=True, axis=1)
    ret['desc'] = df
    return ret


print("preprocessing data ...")

ret = process_data(J(data_path, rd1_train_name))
ret['desc'].to_csv(J(out_path, "rd1_train.csv"))
ret['label'].to_csv(J(out_path, "rd1_train_label.csv"))

ret = process_data(J(data_path, rd1_testA_name))
ret['desc'].to_csv(J(out_path, "rd1_testA.csv"))

ret = process_data(J(data_path, rd1_testB_name))
ret['desc'].to_csv(J(out_path, "rd1_testB.csv"))

ret = process_data(J(data_path, rd2_train_name))
ret['desc'].to_csv(J(out_path, "rd2_train.csv"))
ret['label'].to_csv(J(out_path, "rd2_train_label.csv"))
ret['label2'].to_csv(J(out_path, "rd2_train_label2.csv"))

try:

    ret = process_data(J(data_path, rd2_testA_name))
    ret['desc'].to_csv(J(out_path, "rd2_testA.csv"))

    ret = process_data(J(data_path, rd2_testB_name))
    ret['desc'].to_csv(J(out_path, "rd2_testB.csv"))

except:
    pass

print("save data to `data/`")