import pandas as pd 
import numpy as np 
import os
from os.path import join as J 
from utils.misc import cvt_fmt


path = "."

num_row = 5000

P1 = [
        0.1442,
        0.09335,
        0.06725,
        0.0264,
        0.1453,
        0.02255,
        0.01325,
        0.14265,
        0.09305,
        0.07105,
        0.02585,
        0.14305,
        0.02195,
        0.01355,
        0.0323,
        0.20985,
        0.0337
 ]

P2 = [
        0.131,
        0.00245,
        0.3576,
        0.00575,
        0.01905,
        0.1097,
        0.0023,
        0.0289,
        0.0087,
        0.0316,
        0.0309,
        0.03205
    ]

try:
    df1 = pd.read_csv(J(path, "result_part1.csv"), index_col=0)
except:
    print("result_part1.csv not exist, using random!")
    df1 = pd.DataFrame( [P1]*num_row, index=range(num_row) )

try:
    df2 = pd.read_csv(J(path, "result_part2.csv"), index_col=0)
except:
    print("result_part2.csv not exist, using random!")
    df2 = pd.DataFrame( [P2]*num_row, index=range(num_row) )

df = pd.concat( [df1, df2], axis=1 )
cvt_fmt(df, J(path, "result.csv"))