import os
import sys
import json
import numpy as np 
from glob2 import glob
from os.path import join as J


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("please give path!")
        sys.exit(-1)

    root_path = sys.argv[1]

    files = glob( J(root_path, "**", "info.json") )

    d = []

    for f in files:
        with open(f) as ff:
            d.append(json.load(ff))

    mean_score = { k : np.mean( [ t[k] for t in d  ] ) for k in d[0].keys() }

    with open( J(root_path, "mean_score.json") , 'w') as f:
        json.dump(mean_score, f, indent=4)