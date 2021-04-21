import torch.nn as nn
from model.bert import BERT
from torch.nn import CrossEntropyLoss
from utils.utils import *


def bert_pretrain_forward(F, model, batch, **kws):
    device = torch.device(F.device)
    return model(batch['bert_input'].to(device))

def bert_pretrain_loss(F, model, batch, forward_barch_fun, **kws):
    device = torch.device(F.device)
    bert_label = batch['bert_label'].to(device)  # (b, L)
    out = forward_barch_fun(F, model, batch, **kws)  # (b, L, n)
    # print(out.shape, bert_label.shape)
    loss = CrossEntropyLoss(ignore_index=0)(out.permute(0,2,1), bert_label)
    n_word = out.shape[-1]
    mask = (bert_label>0).reshape(-1).float().to(device)
    acc = calc_acc(out.reshape(-1, n_word), bert_label.reshape(-1), multi=True, mask=mask)
    sc = {'loss':loss.item(), "acc": acc}
    return loss, sc

@torch.no_grad()
def bert_pretrain_eval(F, model, dl, forward_barch_fun, **kws):
    device = torch.device(F.device)
    label_list = []
    out_list = []
    for b in dl:
        out_list.append( forward_barch_fun(F, model, b, **kws) )
        label_list.append( b['bert_label'].to(device) )

    out = torch.cat(out_list, dim=0)
    label = torch.cat(label_list, dim=0)
    loss = CrossEntropyLoss(ignore_index=0)(out.permute(0,2,1), label)
    n_word = out.shape[-1]
    mask = (label>0).reshape(-1).float().to(device)
    acc = calc_acc(out.reshape(-1, n_word), label.reshape(-1), multi=True, mask=mask)
    sc = {'val_loss': loss.item(), "val_acc": acc}
    return sc


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        # self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label=None):
        x = self.bert(x, segment_label)
        return self.mask_lm(x)  # self.next_sentence(x),


# class NextSentencePrediction(nn.Module):
#     """
#     2-class classification model : is_next, is_not_next
#     """

#     def __init__(self, hidden):
#         """
#         :param hidden: BERT model output size
#         """
#         super().__init__()
#         self.linear = nn.Linear(hidden, 2)
#         # self.softmax = nn.LogSoftmax(dim=-1)

#     def forward(self, x):
#         return self.softmax(self.linear(x[:, 0]))


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
        # self.softmax = nn.LogSoftmax(dim=-1)
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear(self.dp(x))
