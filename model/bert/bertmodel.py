import torch
from .bert import *
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from utils.utils import *


def bert_pretrain_forward(F, model, batch, **kws):
    device = torch.device(F.device)
    return model.forward_mlm(batch['bert_input'].to(device))

def bert_pretrain_loss(F, model, batch, forward_barch_fun, **kws):
    device = torch.device(F.device)
    bert_label = batch['bert_label'].to(device)  # (b, L)
    out = forward_barch_fun(F, model, batch, **kws)  # (b, L, n)
    loss = CrossEntropyLoss(ignore_index=0)(out.permute(0,2,1), bert_label)
    n_word = out.shape[-1]
    mask = (bert_label>0).reshape(-1).float().to(device)
    acc = calc_acc(out.reshape(-1, n_word), bert_label.reshape(-1), multi=True, mask=mask)
    sc = {'loss':loss.item(), "acc": acc.item()}
    return loss, sc

@torch.no_grad()
def bert_pretrain_eval(F, model, dl, forward_barch_fun, **kws):
    device = torch.device(F.device)
    loss = 0
    cnt  = 0
    acc = 0
    for b in dl:
        out = forward_barch_fun(F, model, b, **kws)
        label = b['bert_label'].to(device)
        loss += CrossEntropyLoss(ignore_index=0, reduction='sum')(out.permute(0,2,1), label)
        cnt += (label>0).sum()
        n_word = out.shape[-1]
        mask = (label>0).reshape(-1).float().to(device)
        acc += calc_acc(out.reshape(-1, n_word), label.reshape(-1), multi=True, mask=mask, reduction='sum')
    loss /= cnt
    acc /= cnt
    sc = {'val_loss': loss.item(), "val_acc": acc.item()}
    return sc


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size, dropout=.1):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear(self.dp(x))


class BERT_model(nn.Module):

    def __init__(self,
                    vocab_size,
                    h_size=256,
                    n_layer=8,
                    n_head=8,
                    num_label=17,
                    dropout=.1,
                ):
        super().__init__()
        self.bert = BERT(vocab_size, h_size, n_layer, n_head, dropout)
        self.fc = nn.Linear(h_size, num_label)
        self.dp = nn.Dropout(dropout)
        self.fc2 = nn.Linear(h_size, h_size)

        self.mask_lm = MaskedLanguageModel(h_size, vocab_size, dropout)

    def forward(self, x):
        x = self.bert(x)  # (b, L, h)
        x = torch.tanh(self.fc2(x[:,0]))
        x = self.dp(x)
        return torch.sigmoid(self.fc(x))

    def forward_mlm(self, x):
        x = self.bert(x)
        return self.mask_lm(x)
