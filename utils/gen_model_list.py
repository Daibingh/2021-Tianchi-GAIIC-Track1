import os
import sys
import json
import pandas as pd
from glob2 import glob
from os.path import join as J


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("please give path!")
        sys.exit(-1)

    root_path = sys.argv[1]

    files = glob( J(root_path, "**", "model.pth") )

    d = []

    for f in files:

        conf = os.path.sep.join( f.split(os.path.sep)[:-2] + ['config.json']  ) 
        w = json.load(open(conf, 'r', encoding='utf-8')).get("pred_W", 1)
        d.append([ f, conf, w  ] )

    df = pd.DataFrame(d, columns=['model_path', 'config_path', 'weight'])
    print(df.head())
    print(df.shape)

    df.to_csv(J(root_path, "model_list.csv"))
