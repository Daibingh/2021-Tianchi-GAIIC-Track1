import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from os.path import join as J
import pickle as pk
import torch
import random
import six
import copy
import os
import logging
import datetime, time
import shutil
import uuid
import json
from glob2 import glob
from tensorboardX import SummaryWriter
from sklearn import metrics
import math


class Conf:

    def __init__(self):
        pass


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# def load_model(model, file):
#     new_dict = torch.load(file)
#     old_dict = model.state_dict()
#     old_dict.update(new_dict)
#     model.load_state_dict(old_dict)

def warmup_cosine_lr(epoch, warmup_epoch=5, max_epoch=100):
    if epoch <= warmup_epoch:
        return epoch / warmup_epoch
    else:
        return 0.5 * ( math.cos( (epoch - warmup_epoch) /(max_epoch - warmup_epoch) * math.pi ) + 1 )

def cosine_lr(epoch, offset=5, max_epoch=100):
    if epoch <= offset:
        return 1
    else:
        return 0.5 * ( math.cos( (epoch - offset) /(max_epoch - offset) * math.pi ) + 1 )

def warmup_step_lr(epoch, warmup_epoch=5, step_size=10, gamma=.1):
    if epoch <= warmup_epoch:
        return epoch / warmup_epoch
    else:
        return gamma ** (epoch // step_size)

def warmup_mstep_lr(epoch, warmup_epoch=5, steps=list(range(0, 100, 10)), gamma=.1):
    if epoch <= warmup_epoch:
        return epoch / warmup_epoch
    else:
        return gamma ** len( [ t for t in steps if epoch <= t ] )

def warmup_only(epoch, warmup_epoch=5):
    if epoch <= warmup_epoch:
        return epoch / warmup_epoch
    else:
        return 1

def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_model(model):
    for p in model.parameters():
        p.requires_grad = True

def save_config(F, file):
    with open(file, mode='w', encoding='utf-8') as f:
        json.dump(F.__dict__, f, indent=4)

def show_config(F):
    for k, v in F.__dict__.items():
        print( "{} = {}".format( k, v ) )
        
def load_config(file, F=None):
    if F is None:
        F = Conf()
    with open(file, mode='r', encoding='utf-8') as f:
        d = json.load(f)
    for k,v in d.items():
        if k != 'config_file':
            setattr(F, k, v)
    return F

def rand_from(a, b):
    if isinstance(a, float):
        return np.random.rand()*(b-a)+a
    else:
        return np.random.randint(a, b)

def read_pkl(file):
    with open(file, 'rb') as f:
        return pk.load(f)


def to_pkl(var, file):
    with open(file, mode='wb') as f:
        pk.dump(var, f)

def extract_wv_table(file):
    file2 = file.replace('model', 'table')
    m = read_pkl(file)
    table = dict(zip(m.wv.index2word, m.wv.vectors))
    to_pkl(table, file2)


def calc_acc(pred, true, th=.5, multi=False, mask=None, reduction='mean'):
    assert reduction in ['mean', 'sum']
    if mask is None:
        mask = torch.ones_like(true)
    with torch.no_grad():
        if multi:
            cnt = (pred.argmax(dim=1) == true).float()[mask>0].sum()
        else:
            cnt = ((pred > th).float() == true).float()[mask>0].sum()
        if reduction == 'mean':
            return cnt / mask.sum()
        else:
            return cnt

def calc_auc(pred, target):
    fpr, tpr, thresholds = metrics.roc_curve(target, pred)
    auc = metrics.auc(fpr, tpr)
    if np.isnan(auc): auc = 0.
    return auc

def rmdir(dir):
    file_list = os.listdir(dir)
    if (len(file_list)==0): return
    for f in file_list:
        ff = os.path.join(dir, f)
        if not os.path.isdir(ff):
            open(ff, 'w').close()
            os.remove(ff)
        else:
            rmdir(ff)
    os.rmdir(dir)


def get_name():
    dt = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    return dt.strftime('%Y_%m_%d_%H_%M_%S')


def beijing(sec, what):
    beijing_time = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

logging.Formatter.converter = beijing

class Logger:
    
    def __init__(self, file_name=None, mode='a', verbose=1):
        self._logger = logging.getLogger("Logger")
        self._logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()
        if verbose < 0:
            stream_handler.setLevel(logging.WARNING)
        elif verbose == 0:
            stream_handler.setLevel(logging.INFO)
        else:
            stream_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)
        if file_name:
            file_handler = logging.FileHandler(file_name, mode=mode)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def info(self, msg):
        self._logger.info(msg)

    def debug(self, msg):
        self._logger.debug(msg)

    def warning(self, msg):
        self._logger.warning(msg)
    
    def add_file_handler(self, file_name, mode='a'):
        file_handler = logging.FileHandler(file_name, mode=mode)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

    def clear(self):
        self._logger.handlers.clear()

class ModelSaver:

    def __init__(self, save_path='.', mode='best', num_best=1, every_epochs=5):
        assert mode in ['best', 'cycle']
        self.save_path = save_path
        self.mode = mode
        self.num_best = num_best
        self.every_epochs = every_epochs
        self.costs = {}
        self.disabled = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if len(self.costs) > 0:
            pd.Series(self.costs, name='cost').sort_values().to_csv(os.path.join(self.save_path,'info.csv'), header=True, index=True)

    def save_model(self, models, folder, info=None, ignore_keys=None):
        assert type(models) == dict
        path = os.path.join(self.save_path, folder)
        if not os.path.exists(path): os.mkdir(path)
        for name, model in models.items():
            d = model.state_dict()
            if ignore_keys is not None:
                for k in ignore_keys:
                    if k in d: d.pop(k)
            torch.save(d, '{}.pth'.format(os.path.join(path, name)))
        if info is not None:
            with open(os.path.join(path, 'info.json'), mode='w', encoding='utf-8') as f:
                json.dump(info, f, indent=2)

    def check(self, models, cost=None, epoch=None, info=None, ignore_keys=None):
        if self.disabled:
            return
        if self.mode == 'best':
            assert cost is not None
            if len(self.costs) < self.num_best:
                folder_name = get_name()
                self.costs[folder_name] = cost
                self.save_model(models, folder_name,  info, ignore_keys)
            else:
                if (cost < np.asarray(list(self.costs.values()))).any():
                    old_folder_name = max(self.costs, key=self.costs.get)
                    self.costs.pop(old_folder_name)
                    # shutil.rmtree(os.path.join(self.save_path, old_folder_name))
                    rmdir(os.path.join(self.save_path, old_folder_name))
                    folder_name = get_name()
                    self.costs[folder_name] = cost
                    self.costs = {k: v for k, v in sorted(self.costs.items(), key=lambda item: item[1])}
                    self.save_model(models, folder_name, info, ignore_keys)
        else:
            assert epoch is not None
            if epoch % self.every_epochs == 0:
                folder_name = get_name()
                self.save_model(models, folder_name, info, ignore_keys)
                self.costs[folder_name] = cost

def get_saver(save_path='.', mode='best', num_best=1, every_epochs=5):
    return ModelSaver(save_path=save_path, mode=mode, num_best=num_best, every_epochs=every_epochs)


class DataLogger:

    def __init__(self, logdir='.'):
        self.writer = None
        self.disabled = False
        self.logdir = logdir

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write(self, data, step):
        if self.disabled: return
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.logdir)
        for k, v in data.items():
            self.writer.add_scalar(k, v, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def get_logger(logdir='.'):
    return DataLogger(logdir=logdir)


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x