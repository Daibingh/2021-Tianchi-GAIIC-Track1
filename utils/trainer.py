import os
import sys
from .utils import *
from .misc import scoring
import copy
import torch
import numpy as np 
import pandas as pd 
import json
import argparse
from os.path import join as J
from torch.nn import BCELoss


def _forward_batch(F, model, batch, **kws):
    device = torch.device(F.device)
    x = batch['desc'].to(device)
    return model(x)


def _get_loss(F, model, batch, forward_barch_fun, **kws):
    device = torch.device(F.device)
    y = batch['label'].to(device)
    y_ = forward_barch_fun(F, model, batch, **kws)
    loss = BCELoss()(y_, y) 
    sc = {}
    sc['loss'] = loss.item()
    sc['auc'] = np.mean( [calc_auc( y_.cpu().detach().numpy()[:,i], y.cpu().detach().numpy()[:,i] )  for i in range(y.shape[1])] )
    sc['score'] = scoring(sc['loss'])
    return loss, sc


def _eval_model(F, model, dl, forward_barch_fun, **kws):
    device = torch.device(F.device)
    score = {}
    with torch.no_grad():
        y_list = []
        y_list_ = []
        for batch in dl:
            y = batch['label'].to(device)
            y_ = forward_barch_fun(F, model, batch, **kws)
            y_list.append(y)
            y_list_.append(y_)
        y = torch.cat(y_list)
        y_ = torch.cat(y_list_)
        loss = BCELoss(reduction='none')(y_, y).cpu().numpy().mean(axis=0)
        score['val_loss'] = float(np.mean(loss))
        auc = [calc_auc( y_.cpu().numpy()[:,i], y.cpu().numpy()[:,i] )  for i in range(y.shape[1])]
        score['val_auc'] = np.mean(auc)
        score['val_score'] = scoring(score['val_loss'])
        for i in range(loss.shape[0]):
            score['val_score_'+str(i)] = scoring(loss[i])
    return score


def _train_step(F, model, optimizer, batch, forward_barch_fun, get_loss_fun, **kws):
    loss, sc = get_loss_fun(F, model, batch, forward_barch_fun, **kws)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss, sc


class Trainer:

    def __init__(self):
        self.best_score = None
        self.best_epoch = None
        self.best_model = None

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()

        parser.add_argument('--info', default='None')
        parser.add_argument('--start_epoch', default=1, type=int)
        parser.add_argument('--epochs', default=5, type=int)
        parser.add_argument('--batch_size', default=5, type=int)
        parser.add_argument('--optimizer', default="SGD")
        parser.add_argument('--lr', default=.001, type=float)
        parser.add_argument('--momentum', default=.9, type=float)
        parser.add_argument('--weight_decay', default=1e-4, type=float)
        parser.add_argument('--workers', default=4, type=int)
        parser.add_argument('--dataset_splits', default=[7,1,2], type=float, nargs='+')
        parser.add_argument('--shuffle_dataset', action="store_true")
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
        parser.add_argument('--not_save_keys_file', default=None)
        parser.add_argument('--primary_score', default="val_loss")
        parser.add_argument('--higher_better', action='store_true')
        parser.add_argument('--early_stop', action='store_true')
        parser.add_argument('--early_stop_num', default=5, type=int)
        parser.add_argument('--folder_id', default=None)
        parser.add_argument('--resume_path', default=None)
        parser.add_argument('--debug', action='store_true')
        parser.add_argument("-f", '--config_file')

        return parser

    def train(self, 
                F, 
                model, 
                dl_tr, 
                dl_val=None,
                forward_batch_fun=None,
                get_loss_fun=None,
                eval_fun=None,
                step_fun=None,
                hold_best_model=False,
                optimizer=None,
                verbose=0,
                stop_cond=None,
                lr_scheduler=None,
                **kws
             ):

        if lr_scheduler is not None:
            assert optimizer is not None

        if forward_batch_fun is None: forward_batch_fun = _forward_batch
        if get_loss_fun is None: get_loss_fun = _get_loss
        if eval_fun is None: eval_fun = _eval_model
        if step_fun is None: step_fun = _train_step

        old_flag =  copy.deepcopy(F.__dict__)

        dl_val = dl_tr if dl_val is None else dl_val

        ignore_keys = None
        if F.not_save_keys_file is not None:
            with open(F.not_save_keys_file, 'r') as f:
                ignore_keys = [t.strip() for t in f.readlines() if len(t.strip())>0]

        if optimizer is not None:
            F.optimizer = optimizer.__class__.__name__
        else:
            if F.optimizer.lower() == "sgd":
                optimizer = torch.optim.SGD(lr=F.lr, params=model.parameters(), momentum=F.momentum, weight_decay=F.weight_decay)
            elif F.optimizer.lower() == "adam":
                optimizer = torch.optim.Adam(lr=F.lr, params=model.parameters(), weight_decay=F.weight_decay)
            elif F.optimizer.lower() == "adamw":
                optimizer = torch.optim.AdamW(lr=F.lr, params=model.parameters(), weight_decay=F.weight_decay)
            else:
                print("optimizer not found or not support!")
                sys.exit(-1)

        if F.resume_path is not None:
            optimizer.load_state_dict(torch.load(J(F.resume_path, "optimizer.pth")))
            model.load_state_dict(torch.load(J(F.resume_path, "model.pth")), strict=False)
            F.start_epoch = json.load(open(J(F.resume_path, "info.json"), mode='r', encoding='utf-8'))['epoch'] + 1
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict( torch.load( J(F.resume_path, "lr_scheduler.pth") ) )

        L = Logger(verbose=verbose)
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
            save_config(F, J(F.saving_path, 'config.json'))

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
                for it, batch in enumerate(dl_tr):
                    loss, sc = step_fun(F, model, optimizer, batch, forward_batch_fun, get_loss_fun, **kws)

                    if lr_scheduler is not None and lr_scheduler.__class__.__name__ == "CosineAnnealingWarmRestarts":
                        lr_scheduler.step( epoch-1 + it/len(dl_tr) )
                        sc['lr'] = lr_scheduler.get_lr()[0]

                    L.info("[{}/{}][{}/{}] - {}".format(epoch, F.epochs, it+1, len(dl_tr), " - ".join( [ "{}: {:.3f}".format(k, v) for k,v in sc.items() ] ) ))
                    L2.write(data=sc, step=(epoch - 1) * len(dl_tr) + it + 1)
                
                model.eval()
                score = eval_fun(F, model, dl_val, forward_batch_fun, **kws)

                if lr_scheduler is not None and lr_scheduler.__class__.__name__ != "CosineAnnealingWarmRestarts":
                    if lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                        lr_scheduler.step( score[F.primary_score] )
                    else:
                        lr_scheduler.step()
                    score['lr'] = lr_scheduler.get_lr()[0]

                L.info("[{}/{}][{}/{}] - {}".format(epoch, F.epochs, len(dl_tr), len(dl_tr),  " - ".join( [ "{}: {:.3f}".format(k, v) for k,v in score.items() ] )  ))
                L2.write(data=score, step=epoch * len(dl_tr))
                save_state = {
                                F.save_model_name: model, 
                                'optimizer': optimizer
                            }
                if lr_scheduler is not None:
                    save_state['lr_scheduler'] = lr_scheduler
                save_info = {
                                'epoch': epoch,
                                **score
                            }
                S.check( save_state,
                        cost=-score[F.primary_score] if F.higher_better else score[F.primary_score],
                        epoch=epoch,
                        info=save_info,
                        ignore_keys=ignore_keys,
                        )
                if F.enable_saving and F.save_last:
                    S.save_model( save_state,  
                                    "last",
                                    info=save_info, 
                                    ignore_keys=ignore_keys,
                                )
                
                if F.higher_better and score[F.primary_score] > _best or not F.higher_better and score[F.primary_score] < _best:
                    _best = score[F.primary_score]
                    _num = 0
                    _best_epoch = epoch
                    self.best_score = score
                    self.best_epoch = _best_epoch
                    if hold_best_model:
                        self.best_model = copy.deepcopy(model)
                else:
                    _num += 1
                if F.early_stop and _num == F.early_stop_num:
                    L.info('>>>>>>>> early stop on {}, the best is {} <<<<<<<<'.format(_best_epoch, _best))
                    break
                if stop_cond is not None and stop_cond(score):
                    L.info('>>>>>>>> cond stop on {}, the best is {} <<<<<<<<'.format(_best_epoch, _best))
                    break

        L.clear()
        for k,v in old_flag.items():
            setattr(F, k, v) 


if __name__ == "__main__":

    pass