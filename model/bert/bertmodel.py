import torch
# from .bert import *
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from utils.utils import *


# def bert_pretrain_forward(F, model, batch, **kws):
#     device = torch.device(F.device)
#     return model.forward_mlm(batch['bert_input'].to(device))

def bert_pretrain_loss(F, model, batch, forward_barch_fun, **kws):
    device = torch.device(F.device)
    input_ids = batch['bert_input'].to(device)
    attention_mask = (input_ids>0).long()
    labels = batch['bert_label'].to(device)  # (b, L)
    labels[labels==0] = -100
    loss = model(input_ids, attention_mask=attention_mask, labels=labels)[0]
    sc = { 'loss':loss.item() }
    return loss, sc

@torch.no_grad()
def bert_pretrain_eval(F, model, dl, forward_barch_fun, **kws):
    device = torch.device(F.device)
    loss = 0.
    for b in dl:
        input_ids = b['bert_input'].to(device)
        attention_mask = (input_ids>0).long()
        labels = b['bert_label'].to(device)  # (b, L)
        labels[labels==0] = -100
        loss += model(input_ids, attention_mask=attention_mask, labels=labels)[0]
    loss /= len(dl)
    sc = { 'val_loss': loss.item() }
    return sc

def bert_forward(F, model, batch, **kws):
    device = torch.device(F.device)
    input_ids = batch['desc'].to(device)
    attention_mask = (input_ids>0).long()
    return torch.sigmoid(model(input_ids, attention_mask=attention_mask)[0])


# class MaskedLanguageModel(nn.Module):
#     """
#     predicting origin token from masked input sequence
#     n-class classification problem, n-class = vocab_size
#     """

#     def __init__(self, hidden, vocab_size, dropout=.1):
#         """
#         :param hidden: output size of BERT model
#         :param vocab_size: total vocab size
#         """
#         super().__init__()
#         self.linear = nn.Linear(hidden, vocab_size)
#         self.dp = nn.Dropout(dropout)

#     def forward(self, x):
#         return self.linear(self.dp(x))


# class BERT_model(nn.Module):

#     def __init__(self,
#                     vocab_size,
#                     h_size=256,
#                     n_layer=8,
#                     n_head=8,
#                     num_label=17,
#                     dropout=.1,
#                 ):
#         super().__init__()
#         self.bert = BERT(vocab_size, h_size, n_layer, n_head, dropout)
#         self.fc = nn.Linear(h_size, num_label)
#         self.dp = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(h_size, h_size)

#         self.mask_lm = MaskedLanguageModel(h_size, vocab_size, dropout)

#     def forward(self, x):
#         x = self.bert(x)  # (b, L, h)
#         x = torch.tanh(self.fc2(x[:,0]))
#         x = self.dp(x)
#         return torch.sigmoid(self.fc(x))

#     def forward_mlm(self, x):
#         x = self.bert(x)
#         return self.mask_lm(x)
