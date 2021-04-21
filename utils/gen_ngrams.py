import numpy as np 
import pandas as pd 


N = 4 

def func(n, text):
    l = text.strip().split()
    g = []
    for i in range(0, len(l)-n+1):
        g.append('-'.join(l[i:i+n]))
    return ' '.join(g)
    

if __name__ == "__main__":

    train_df = pd.read_csv("data/rd1_train.csv", index_col=0)
    test_df = pd.read_csv('data/rd1_testA.csv', index_col=0)

    train_rpt_ng = train_df.desc.map(lambda t: func(N, t))
    test_rpt_ng = test_df.desc.map( lambda t: func( N, t ) )

    train_rpt_ng.to_csv('data/rd1_train_{}g.csv'.format(N), header=True)
    test_rpt_ng.to_csv('data/rd1_testA_{}g.csv'.format(N), header=True)