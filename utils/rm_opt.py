import os
import sys
from glob2 import glob
from os.path import join as J


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("please give path!")
        sys.exit(-1)

    root_path = sys.argv[1]

    files = glob( J(root_path, "**", "optimizer.pth" ) ) \
          + glob( J(root_path, "**", "lr_scheduler.pth" ) )

    for f in files:
        print(f)
        os.remove(f)