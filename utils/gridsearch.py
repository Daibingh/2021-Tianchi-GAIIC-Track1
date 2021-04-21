import warnings
warnings.filterwarnings('ignore')
import copy
import os 
from .utils import *
from .trainer import *
import numpy as np 
import pandas as pd 
from os.path import join as J


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def grid_search(F, 
                T,
                model, 
                dl_tr,
                dl_val,
                params,
                fast=True,
                out_path='.',
                verbose=-1,
                **kws,
                ):

    assert type(params) == dict
    for k in params:
        assert k in F.__dict__

    F.enable_logging = False
    F.enable_saving = False

    mid_params = { k : np.median(v)  for k, v in params.items() }

    best_params = copy.deepcopy(mid_params)

    curr_params = copy.deepcopy(mid_params)

    for k, v in mid_params.items():
        setattr(F, k, v)
    
    recd = []
    _best = -np.inf if F.higher_better else np.inf
    for key, values in params.items():

        for v in values:

            print(bcolors.OKGREEN + "****** training when {} = {}".format(key, v) + bcolors.ENDC)

            curr_params[key] = v
            setattr(F, key, v)

            T.train(
                F,
                model,
                dl_tr,
                dl_val,
                verbose=verbose,
                **kws,
                )

            curr_params['score'] = T.best_score[F.primary_score]
            recd.append(copy.deepcopy(curr_params))

            print(bcolors.OKGREEN + ">>>>>> finish when {} = {}, best = {}".format(key, v, curr_params['score']) + bcolors.ENDC)

            if ( F.higher_better and T.best_score[F.primary_score] > _best ) or ( not F.higher_better and T.best_score[F.primary_score] < _best ):
                _best = T.best_score[F.primary_score]
                best_params[key] = v
                
        curr_params[key] = best_params[key]
        setattr(F, key, best_params[key])

    print("="*60)
    for k, v in best_params.items():
        print(bcolors.OKCYAN + "{} = {}".format(k, v) + bcolors.ENDC)
    print(bcolors.OKCYAN + "best = {}".format(_best) + bcolors.ENDC)

    recd = pd.DataFrame( recd ).sort_values('score', ascending= not F.higher_better).reset_index( drop=True)
    recd.to_csv( J(out_path, "{}_record.csv".format(get_name())) )

