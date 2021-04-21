import json
from glob2 import glob
import os
import sys
from os.path import join as J


updated = {
    "wv_model_file": "data/wv_model_128.pkl",
}


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("please give path!")
        sys.exit(-1)

    root_path = sys.argv[1]
    files = glob( J( root_path, "**", "*.json" ) )

    for f in files:
        with open(f, 'r', encoding='utf-8') as ff:
            try:
                conf = json.load(ff)
            except Exception as e:
                print(e)
                continue
        
        for k, v in updated.items():
            if k in conf:
                print("CHANGE {}, {}: {} -> {}".format(f, k, conf[k], v))
                conf[k] = v

        with open(f, 'w', encoding='utf-8') as ff:
            json.dump(conf, ff, indent=4)