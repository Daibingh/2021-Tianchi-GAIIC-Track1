import pandas as pd
import numpy as np


files = [
    "data/rd1_train.csv",
    "data/rd2_train.csv",
    "data/rd1_testA.csv",
    "data/rd1_testB.csv",
]

text_list = []

for f in files:
    df = pd.read_csv(f, index_col=0)
    text_list += df.desc.tolist()

print("num of lines:", len(text_list))

text = '\n'.join( text_list ) 

with open("data/corpus.txt", 'w') as f:
    f.write(text)
