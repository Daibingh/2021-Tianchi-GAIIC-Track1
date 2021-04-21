import torch
from torch.utils.data import DataLoader
from os.path import join as J
from model import *
from baselines import *
import pandas as pd
from torch.nn import BCELoss
import argparse
from utils import *
import os
import json


def crt_model(F):
    pass

def crt_dataset(F):
    pass

def get_loss(model, batch):
    loss = 0.

    return loss

def eval_model(model, dl):
    score = {}
    with torch.no_grad():

        for data in dl:
            pass
            
    return score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--info', default='None')
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--optimizer', default="SGD")
    parser.add_argument('--lr', default=.001, type=float)
    parser.add_argument('--step_lr', action='store_true')
    parser.add_argument('--step_size', default=10, type=int)
    parser.add_argument('--gamma', default=.5, type=float)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--dataset_splits', default="[7,1,2]")
    parser.add_argument('--random_seed', default=1212, type=int)
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--enable_logging', action='store_true')
    parser.add_argument('--enable_saving', action='store_true')
    parser.add_argument('--logging_path', default="logging")
    parser.add_argument('--saving_path', default="saving")
    parser.add_argument('--save_num_best', default=1, type=int)
    parser.add_argument('--save_mode', default="best")
    parser.add_argument('--save_model_name', default="model")
    parser.add_argument('--save_every_epochs', default=10, type=int)
    parser.add_argument('--save_last', action='store_true')
    parser.add_argument('--primary_score', default="val_loss")
    parser.add_argument('--higher_better', action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--early_stop_num', default=5, type=int)
    parser.add_argument('--folder_id', default=None)
    parser.add_argument('--resume_path', default=None)

    F = parser.parse_args()

    setup_seed(F.random_seed)
    device = torch.device(F.device)
    F.dataset_splits = eval(F.dataset_splits)

    dataset = crt_dataset(F)    
    dataset_train, dataset_eval, dataset_test = dataset.split(F.dataset_splits, shuffle=True)

    dl_tr = DataLoader( dataset_train,
        batch_size=F.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=F.workers,
    )

    dl_eval = DataLoader( dataset_eval,
        batch_size=F.batch_size*4,
        drop_last=False,
        shuffle=True,
        num_workers=F.workers,
    )

    if len(dataset_test) > 0:
        dl_test = DataLoader( dataset_test,
            batch_size=F.batch_size*4,
            drop_last=False,
            shuffle=True,
            num_workers=F.workers,
        )


    model = crt_model(F)
    
    if F.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(lr=F.lr, params=model.parameters(), momentum=.9, weight_decay=1e-4)
    elif F.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(lr=F.lr, params=model.parameters())
    else:
        optimizer = torch.optim.SGD(lr=F.lr, params=model.parameters(), momentum=.9, weight_decay=1e-4)

    if F.resume_path is not None:
        optimizer.load_state_dict(torch.load(J(F.resume_path, "optimizer.pth")))
        model.load_state_dict(torch.load(J(F.resume_path, "model.pth")))
        F.start_epoch = json.load(open(J(F.resume_path, "info.json"), mode='r', encoding='utf-8'))['epoch'] + 1

    if F.step_lr and F.optimizer.lower() != "adam":
        StepLR = torch.optim.lr_scheduler.StepLR(optimizer, F.step_size, gamma=F.gamma, last_epoch=F.start_epoch-2)

    L = Logger()
    if F.folder_id is None:
        F.folder_id = "={}=".format(get_name())
    else:
        F.folder_id = "={}_{}=".format(get_name(), F.folder_id)
    if F.resume_path is not None:
        logging_path = J( F.logging_path, "={}=".format(F.resume_path.split('=')[1]) )
    else:
        logging_path = J(F.logging_path, F.folder_id)
    if F.enable_logging:
        if not os.path.exists(logging_path):
            os.mkdir(logging_path)
        L.add_file_handler(J(logging_path, 'log.txt'), mode='a')
    
    if F.resume_path is not None:
        saving_path = os.path.split(F.resume_path)[0]
    else:
        saving_path = J(F.saving_path, F.folder_id)
    if F.enable_saving and not os.path.exists(saving_path):
        os.mkdir(saving_path)

    F.logging_path = logging_path
    F.saving_path = saving_path

    for k, v in F.__dict__.items():
        L.info("{} = {}".format(k, v))

    if F.enable_saving:
        with open(J(F.saving_path, 'config.json'), mode='w', encoding='utf-8') as f:
                json.dump(F.__dict__, f, indent=2)

    with get_logger(logging_path) as L2, \
                get_saver(saving_path, num_best=F.save_num_best, mode=F.save_mode,
                          every_epochs=F.save_every_epochs) as S:
        L2.disabled = not F.enable_logging
        S.disabled = not F.enable_saving
        _best = -np.inf if F.higher_better else np.inf
        _num = 0
        _best_epoch = 1
        for epoch in range(F.start_epoch, F.epochs+1):
            model.train()
            sc = {}
            for it, batch in enumerate(dl_tr):
                loss = get_loss(model, batch)                
                sc['loss'] = loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                L.info("[{}/{}][{}/{}] - {}".format(epoch, F.epochs, it+1, len(dl_tr), " - ".join( [ "{}: {:.3f}".format(k, v) for k,v in sc.items() ] ) ))
                L2.write(data=sc, step=(epoch - 1) * len(dl_tr) + it + 1)
            
            if F.step_lr and F.optimizer.lower() != "adam":
                StepLR.step()
            model.eval()
            score = eval_model(model, dl_eval, mask=label_mask)
            L.info("[{}/{}][{}/{}] - {}".format(epoch, F.epochs, len(dl_tr), len(dl_tr),  " - ".join( [ "{}: {:.3f}".format(k, v) for k,v in score.items() ] )  ))
            L2.write(data=score, step=epoch * len(dl_tr))
            S.check({F.save_model_name: model, 'optimizer': optimizer},
                            cost=-score[F.primary_score] if F.higher_better else score[F.primary_score],
                            epoch=epoch,
                            info={'epoch': epoch,
                                  **score
                                  }
                            )
            if F.enable_saving and F.save_last:
                S.save_model( { F.save_model_name: model, 'optimizer': optimizer },  
                                "last",
                                info={ 'epoch': epoch, **score }, 
                            )
            if F.early_stop:
                if F.higher_better and score[F.primary_score] > _best or not F.higher_better and score[F.primary_score] < _best:
                    _best = score[F.primary_score]
                    _num = 0
                    _best_epoch = epoch
                else:
                    _num += 1
                if _num == F.early_stop_num:
                    L.info('>>>>>>>> early stop on {}, the best is {} <<<<<<<<'.format(_best_epoch, _best))
                    break
