from .modeling_nezha import BertForMaskedLM, BertForSequenceClassification
import torch
from utils.utils import *
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCELoss


def nezha_pretrain_loss(F, model, batch, forward_barch_fun, **kws):

    device = torch.device(F.device)
    input_ids = batch['bert_input'].to(device)
    attention_mask = (input_ids>0).long()
    masked_lm_labels = batch['bert_label'].to(device)  # (b, L)
    loss = model(input_ids, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels)
    sc = { 'loss':loss.item() }
    return loss, sc

@torch.no_grad()
def nezha_pretrain_eval(F, model, dl, forward_barch_fun, **kws):
    device = torch.device(F.device)
    loss = 0.
    for b in dl:
        input_ids = b['bert_input'].to(device)
        attention_mask = (input_ids>0).long()
        masked_lm_labels = b['bert_label'].to(device)  # (b, L)
        loss += model(input_ids, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels)
    loss /= len(dl)
    sc = { 'val_loss': loss.item() }
    return sc


def nezha_forward(F, model, batch, **kws):
    device = torch.device(F.device)
    input_ids = batch['desc'].to(device)
    attention_mask = (input_ids>0).long()
    return model(input_ids, attention_mask=attention_mask)


def train_step_with_fgm(F, model, optimizer, batch, forward_barch_fun, get_loss_fun, **kws):
    
    emb_name = kws.get("emb_name", "word_embeddings")
    eps = kws.get("fgm_eps", 0.1)

    loss, sc = get_loss_fun(F, model, batch, forward_barch_fun)
    loss.backward()
    backup = None
    for name, param in model.named_parameters():
        if param.requires_grad and emb_name in name:
            norm = torch.norm(param.grad)
            if norm != 0 and not torch.isnan(norm):
                backup = param.data.clone()
                r_at = eps * param.grad / norm
                param.data.add_(r_at)
                break
    if backup is not None:
        loss, sc = get_loss_fun(F, model, batch, forward_barch_fun)
        sc = { k+"_adv" : v for k,v in sc.items() }
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and emb_name in name: 
                param.data = backup
                break
    optimizer.step()
    optimizer.zero_grad()
    return loss, sc


# def nezha_loss(F, model, batch, forward_barch_fun, **kws):

#     device = torch.device(F.device)
#     pred = forward_barch_fun(F, model, batch, **kws)
#     label = batch['label'].to(device)
#     loss = BCELoss()(pred, label)
#     sc = { 'loss':loss.item() }
#     return loss, sc

# @torch.no_grad()
# def bert_eval(F, model, dl, forward_barch_fun, **kws):
#     device = torch.device(F.device)
#     loss = 0.
#     for b in dl:
#         input_ids = b['bert_input'].to(device)
#         attention_mask = (input_ids>0).long()
#         masked_lm_labels = b['bert_label'].to(device)  # (b, L)
#         loss += model(input_ids, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels)
#     loss /= len(dl)
#     sc = { 'val_loss': loss.item() }
#     return sc