import json
from glob2 import glob
import os
import sys
from os.path import join as J


ignore = {
    "wv_model_file",
    "logging_path",
    "saving_path",
    "folder_id",
    "fold_start",
}


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("args: dst src")
        sys.exit(-1)

    dst = sys.argv[1]
    path = sys.argv[2]

    assert '.json' not in path

    src = glob(J(path, "**", "config.json"))[0]

    print("SRC:", src)
    print("DST:", dst)

    with open(src, 'r', encoding='utf-8') as f:
        conf_src = json.load(f)

    with open(dst, 'r', encoding='utf-8') as f:
        conf_dst = json.load(f)

    for k, v in conf_src.items():
        if k in conf_dst and k not in ignore and v!= conf_dst[k]:
            print("CHANGE {}: {} -> {}".format(k, conf_dst[k], v))
            conf_dst[k] = v

    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(conf_dst, f, indent=4)