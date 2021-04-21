import os
import sys
from glob2 import glob
from os.path import join as J
import shutil
import pandas as pd 


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("please give path and n!")
        sys.exit(-1)

    root_path = sys.argv[1]
    n = int(sys.argv[2])

    files = glob( J(root_path, "**", "info.csv" ) ) 

    for f in files:
        df = pd.read_csv(f)
        fds = df.iloc[:,0].tolist()
        if len(fds) <= n: continue
        path = os.path.split(f)[0]
        for t in fds[n:]:
            fd = J(path, t)
            print("REMOVE", fd)
            try:
                shutil.rmtree(fd)
            except Exception as e:
                print(e)