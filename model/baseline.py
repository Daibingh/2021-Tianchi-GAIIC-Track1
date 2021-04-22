import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd 
from utils.utils import *
from .transformer import TransformerEncoderLayer as TFE


class MaskLSTM(nn.Module):
    """
    one layer LSTM with mask, support bidirection.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

    def _zero_init(self, batch_size):
        h = torch.zeros(batch_size, self.hidden_size, device=next(self.parameters()).device)
        c = torch.zeros(batch_size, self.hidden_size, device=next(self.parameters()).device)
        return h, c

    def forward(self, x, mask=None, mask_zero=False, hc=None, bid=False):
        """
        :param x: shape: batch x len x dim
        :param mask: shape: batch x len
        :param hc: tuple (h, c) shape: (batch x dim, batch x dim)
        :return: batch x len x dim, (batch x dim, batch x dim)
        """
        device = next(self.parameters()).device
        if hc is None:
            h, c = self._zero_init(batch_size=x.shape[0])
        if mask is None:
            if mask_zero:
                mask = (x.sum(dim=-1) > 0).float().unsqueeze(2)
            else:
                mask = torch.ones(x.shape[0], x.shape[1], 1, device=device)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(2)
        if bid:
            out = torch.zeros(x.shape[0], x.shape[1], self.hidden_size*2, device=device)
            h_n = torch.zeros(x.shape[0], self.hidden_size*2, device=device)
            c_n = torch.zeros(x.shape[0], self.hidden_size*2, device=device)
        else:
            out = torch.zeros(x.shape[0], x.shape[1], self.hidden_size, device=device)
            h_n = torch.zeros(x.shape[0], self.hidden_size, device=device)
            c_n = torch.zeros(x.shape[0], self.hidden_size, device=device)
        for i in range(x.shape[1]):
            h1, c1 = self.lstm(x[:, i, :], (h, c))
            h1 = mask[:, i] * h1 + (1 - mask[:, i]) * h
            c1 = mask[:, i] * c1 + (1 - mask[:, i] * c)
            h, c = h1, c1
            out[:, i, :self.hidden_size] = h
        h_n[:, :self.hidden_size] = h
        c_n[:, :self.hidden_size] = c
        if bid:
            for i in range(x.shape[1]-1, -1, -1):
                h1, c1 = self.lstm(x[:, i, :], (h, c))
                h1 = mask[:, i] * h1 + (1 - mask[:, i]) * h
                c1 = mask[:, i] * c1 + (1 - mask[:, i] * c)
                h, c = h1, c1
                out[:, i, self.hidden_size:] = h
                
            h_n[:, self.hidden_size:] = h
            c_n[:, self.hidden_size:] = c
           
        return out, (h_n, c_n)


class MaskGRU(nn.Module):
    """
    one layer GRU with mask, support bidirection.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)

    def _zero_init(self, batch_size):
        h = torch.zeros(batch_size, self.hidden_size, device=next(self.parameters()).device)
        return h

    def forward(self, x, mask=None, mask_zero=False, h=None, bid=False):
        """
        :param x: shape: batch x len x dim
        :param mask: shape: batch x len
        :param hc: tuple (h, c) shape: (batch x dim, batch x dim)
        :return: batch x len x dim, (batch x dim, batch x dim)
        """
        device = next(self.parameters()).device
        if h is None:
            h = self._zero_init(batch_size=x.shape[0])
        if mask is None:
            if mask_zero:
                mask = (x.sum(dim=-1) > 0).float().unsqueeze(2)
            else:
                mask = torch.ones(x.shape[0], x.shape[1], 1, device=device)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(2)
        if bid:
            out = torch.zeros(x.shape[0], x.shape[1], self.hidden_size*2, device=device)
            h_n = torch.zeros(x.shape[0], self.hidden_size*2, device=device)
        else:
            out = torch.zeros(x.shape[0], x.shape[1], self.hidden_size, device=device)
            h_n = torch.zeros(x.shape[0], self.hidden_size, device=device)
        for i in range(x.shape[1]):
            h1 = self.gru(x[:, i, :], h)
            h1 = mask[:, i] * h1 + (1 - mask[:, i]) * h
            h = h1
            out[:, i, :self.hidden_size] = h
        h_n[:, :self.hidden_size] = h
        if bid:
            for i in range(x.shape[1]-1, -1, -1):
                h1 = self.gru(x[:, i, :], h)
                h1 = mask[:, i] * h1 + (1 - mask[:, i]) * h
                h = h1
                out[:, i, self.hidden_size:] = h
                
            h_n[:, self.hidden_size:] = h
           
        return out, h_n


class Baseline(nn.Module):
    """baseline model: one-layer single direction LSTM

    """
    def __init__(self, 
                emb_size,
                wv_model_file,
                h_size,
                num_label,
                fix_emb=True,
                dropout=0.,
                use_pretrain=True,
                    ):
        super().__init__()
        wv = read_pkl(wv_model_file).wv.vectors
        self.emb = nn.Embedding(num_embeddings=wv.shape[0]+1, embedding_dim=emb_size, padding_idx=0)
        if use_pretrain: 
            wv = np.concatenate( ( np.zeros((1, wv.shape[1])), wv ) )
            self.emb.weight.data.copy_(torch.from_numpy(wv))
            if fix_emb: 
               self.emb.weight.requires_grad = False
        self.lstm = nn.LSTM(emb_size, hidden_size=h_size, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(h_size, num_label)
    
    def forward(self, x):
        x = self.emb(x)
        _, (x, _) = self.lstm(x)
        x = torch.sigmoid(self.fc(x.squeeze(0)))
        return x
        

class LSTM_model(nn.Module):
    """LSTM model support n-layers single or double direction

    """
    def __init__(self, 
                emb_size,
                wv_model_file,
                h_size,
                num_label,
                fix_emb=True,
                dropout=0.,
                use_pretrain=True,
                bidirectional=False,
                num_layer=1,
                    ):
        super().__init__()
        wv = read_pkl(wv_model_file).wv.vectors
        self.emb = nn.Embedding(num_embeddings=wv.shape[0]+1, embedding_dim=emb_size, padding_idx=0)
        if use_pretrain: 
            wv = np.concatenate( ( np.zeros((1, wv.shape[1])), wv ) )
            self.emb.weight.data.copy_(torch.from_numpy(wv))
            if fix_emb: 
               self.emb.weight.requires_grad = False
        self.lstm = nn.LSTM(emb_size, hidden_size=h_size, batch_first=True, dropout=dropout, bidirectional=bidirectional, num_layers=num_layer)
        if bidirectional:
            self.fc = nn.Linear(h_size*2*num_layer, num_label)
        else:
            self.fc = nn.Linear(h_size*num_layer, num_label)
        self.dp = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.emb(x)
        x = self.dp(x)
        _, (x, _) = self.lstm(x)
        x = x.permute(1,0,2).reshape(batch_size, -1)
        x = self.dp(x)
        x = torch.sigmoid(self.fc(x))
        return x


class GRU_model(nn.Module):
    """GRU model, support n-layers single or double direction
    """
    def __init__(self, 
                emb_size,
                wv_model_file,
                h_size,
                num_label,
                fix_emb=True,
                dropout=0.,
                use_pretrain=True,
                bidirectional=False,
                num_layer=1,
                    ):
        super().__init__()
        wv = read_pkl(wv_model_file).wv.vectors
        self.emb = nn.Embedding(num_embeddings=wv.shape[0]+1, embedding_dim=emb_size, padding_idx=0)
        if use_pretrain: 
            wv = np.concatenate( ( np.zeros((1, wv.shape[1])), wv ) )
            self.emb.weight.data.copy_(torch.from_numpy(wv))
            if fix_emb: 
               self.emb.weight.requires_grad = False
        self.gru = nn.GRU(emb_size, hidden_size=h_size, batch_first=True, dropout=dropout, bidirectional=bidirectional, num_layers=num_layer)
        if bidirectional:
            self.fc = nn.Linear(h_size*2*num_layer, num_label)
        else:
            self.fc = nn.Linear(h_size*num_layer, num_label)
        self.dp = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.emb(x)
        x = self.dp(x)
        _, x = self.gru(x)
        x = x.permute(1,0,2).reshape(batch_size, -1)
        x = self.dp(x)
        x = torch.sigmoid(self.fc(x))
        return x


class MaskGRU_model(nn.Module):
    """wrapper for mask gru for classification: one-layer single or double direction
    Args:
        nn ([type]): [description]
    """
    def __init__(self, 
                emb_size,
                wv_model_file,
                h_size,
                num_label,
                fix_emb=True,
                dropout=0.,
                use_pretrain=True,
                bidirectional=False,
                num_layer=1,
                    ):
        super().__init__()
        wv = read_pkl(wv_model_file).wv.vectors
        self.emb = nn.Embedding(num_embeddings=wv.shape[0]+1, embedding_dim=emb_size, padding_idx=0)
        if use_pretrain: 
            wv = np.concatenate( ( np.zeros((1, wv.shape[1])), wv ) )
            self.emb.weight.data.copy_(torch.from_numpy(wv))
            if fix_emb: 
               self.emb.weight.requires_grad = False
        self.gru = MaskGRU(emb_size, h_size)
        if bidirectional:
            self.fc = nn.Linear(h_size*2, num_label)
        else:
            self.fc = nn.Linear(h_size, num_label)
        self.dp = nn.Dropout(dropout)
        self.bid = bidirectional
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.emb(x)
        x = self.dp(x)
        _, x = self.gru(x, bid=self.bid)
        x = self.dp(x)
        x = torch.sigmoid(self.fc(x))
        return x


class GRU_model2(nn.Module):
    """minor differences with GRU_model
    """
    def __init__(self, 
                emb_size,
                wv_model_file,
                h_size,
                num_label,
                fix_emb=True,
                dropout=0.,
                use_pretrain=True,
                bidirectional=False,
                num_layer=1,
                    ):
        super().__init__()
        wv = read_pkl(wv_model_file).wv.vectors
        self.emb = nn.Embedding(num_embeddings=wv.shape[0]+1, embedding_dim=emb_size, padding_idx=0)
        if use_pretrain: 
            wv = np.concatenate( ( np.zeros((1, wv.shape[1])), wv ) )
            self.emb.weight.data.copy_(torch.from_numpy(wv))
            if fix_emb: 
               self.emb.weight.requires_grad = False
        self.gru = nn.GRU(emb_size, hidden_size=h_size, batch_first=True, dropout=dropout, bidirectional=bidirectional, num_layers=num_layer)
        if bidirectional:
            self.fc = nn.Linear(h_size*2, num_label)
        else:
            self.fc = nn.Linear(h_size, num_label)
        self.dp = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.emb(x)
        x, _ = self.gru(x)
        # x = x.permute(1,0,2).reshape(batch_size, -1)
        x = self.dp(x[:, -1, :])
        x = torch.sigmoid(self.fc(x))
        return x


class BiLSTM_Atten(nn.Module):
    """ BiLSTM with standard attention for text MLC
    """

    def __init__(self, 
                emb_size,
                wv_model_file,
                h_size,
                num_label,
                fix_emb=True,
                dropout=0.,
                use_pretrain=True,
                num_layer=1,
        ):

        super().__init__()
        wv = read_pkl(wv_model_file).wv.vectors
        self.emb = nn.Embedding(num_embeddings=wv.shape[0]+1, embedding_dim=emb_size, padding_idx=0)
        if use_pretrain: 
            wv = np.concatenate( ( np.zeros((1, wv.shape[1])), wv ) )
            self.emb.weight.data.copy_(torch.from_numpy(wv))
            if fix_emb: 
               self.emb.weight.requires_grad = False

        self.rnn = nn.LSTM(emb_size, h_size, num_layers=num_layer, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(h_size * 2, num_label)
        self.dropout = nn.Dropout(dropout)

    #x,query：[batch, seq_len, hidden_dim*2]
    def attention_net(self, x, query, mask=None):      #软性注意力机制（key=value=x）

        d_k = query.size(-1)                                              #d_k为query的维度
        scores = torch.matmul(query, x.transpose(1, 2)) / np.sqrt(d_k)  #打分机制  scores:[batch, seq_len, seq_len]

        p_attn = torch.softmax(scores, dim = -1)                              #对最后一个维度归一化得分
        context = torch.matmul(p_attn, x).sum(1)       #对权重化的x求和，[batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, p_attn


    def forward(self, x):
        x = self.emb(x)       #[batch, seq_len,  embedding_dim]

        # output: [ batch, seq_len, hidden_dim*2]     hidden/cell: [n_layers*2, batch, hidden_dim]
        output, (final_hidden_state, final_cell_state) = self.rnn(x)

        query = self.dropout(output)
        attn_output, attention = self.attention_net(output, query)       #和LSTM的不同就在于这一句
        attn_output = self.dropout(attn_output)
        logit = self.fc(attn_output)
        return torch.sigmoid(logit)


class BiGRU_Atten(nn.Module):
    """attention 标准公式
    """

    def __init__(self, 
                emb_size,
                wv_model_file,
                h_size,
                num_label,
                fix_emb=True,
                dropout=0.,
                use_pretrain=True,
                num_layer=1,
        ):

        super().__init__()
        wv = read_pkl(wv_model_file).wv.vectors
        self.emb = nn.Embedding(num_embeddings=wv.shape[0]+1, embedding_dim=emb_size, padding_idx=0)
        if use_pretrain: 
            wv = np.concatenate( ( np.zeros((1, wv.shape[1])), wv ) )
            self.emb.weight.data.copy_(torch.from_numpy(wv))
            if fix_emb: 
               self.emb.weight.requires_grad = False

        self.rnn = nn.GRU(emb_size, h_size, num_layers=num_layer, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(h_size * 2, num_label)
        self.dropout = nn.Dropout(dropout)

    #x,query：[batch, seq_len, hidden_dim*2]
    def attention_net(self, x, query, mask=None):      #软性注意力机制（key=value=x）

        d_k = query.size(-1)                                              #d_k为query的维度
        scores = torch.matmul(query, x.transpose(1, 2)) / np.sqrt(d_k)  #打分机制  scores:[batch, seq_len, seq_len]

        p_attn = torch.softmax(scores, dim = -1)                              #对最后一个维度归一化得分
        context = torch.matmul(p_attn, x).sum(1)       #对权重化的x求和，[batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, p_attn


    def forward(self, x):
        x = self.emb(x)       #[batch, seq_len, embedding_dim]

        # output: [batch, seq_len, hidden_dim*2]     hidden/cell: [n_layers*2, batch, hidden_dim]
        output, _ = self.rnn(x)

        query = self.dropout(output)
        attn_output, attention = self.attention_net(output, query)       #和LSTM的不同就在于这一句
        attn_output = self.dropout(attn_output)
        logit = self.fc(attn_output)
        return torch.sigmoid(logit)


class BiLSTM_Atten2(nn.Module):

    def __init__(self, 
                emb_size,
                wv_model_file,
                h_size,
                num_label,
                fix_emb=True,
                dropout=0.,
                use_pretrain=True,
                num_layer=1,
        ):

        super().__init__()
        wv = read_pkl(wv_model_file).wv.vectors
        self.emb = nn.Embedding(num_embeddings=wv.shape[0]+1, embedding_dim=emb_size, padding_idx=0)
        if use_pretrain: 
            wv = np.concatenate( ( np.zeros((1, wv.shape[1])), wv ) )
            self.emb.weight.data.copy_(torch.from_numpy(wv))
            if fix_emb: 
               self.emb.weight.requires_grad = False

        self.rnn = nn.LSTM(emb_size, h_size, num_layers=num_layer, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(h_size * 2, num_label)
        self.dropout = nn.Dropout(dropout)

        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(h_size * 2, h_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(h_size * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)


    def attention_net(self, x):       #x:[batch, seq_len, hidden_dim*2]

        u = torch.tanh(torch.matmul(x, self.w_omega))         #[batch, seq_len, hidden_dim*2]
        att = torch.matmul(u, self.u_omega)                   #[batch, seq_len, 1]
        att_score = torch.softmax(att, dim=1)

        scored_x = x * att_score                              #[batch, seq_len, hidden_dim*2]

        context = torch.sum(scored_x, dim=1)                  #[batch, hidden_dim*2]
        return context


    def forward(self, x):
        x = self.emb(x)       #[batch, seq_len, embedding_dim]
        x = self.dropout(x)
        # output: [batch, seq_len, hidden_dim*2]     hidden/cell: [n_layers*2, batch, hidden_dim]
        output, (final_hidden_state, final_cell_state) = self.rnn(x)

        attn_output = self.attention_net(output)
        attn_output = self.dropout(attn_output)
        logit = self.fc(attn_output)
        return torch.sigmoid(logit)


class BiGRU_Atten2(nn.Module):

    def __init__(self, 
                emb_size,
                wv_model_file,
                h_size,
                num_label,
                fix_emb=True,
                dropout=0.,
                use_pretrain=True,
                num_layer=1,
        ):

        super().__init__()
        wv = read_pkl(wv_model_file).wv.vectors
        self.emb = nn.Embedding(num_embeddings=wv.shape[0]+1, embedding_dim=emb_size, padding_idx=0)
        if use_pretrain: 
            wv = np.concatenate( ( np.zeros((1, wv.shape[1])), wv ) )
            self.emb.weight.data.copy_(torch.from_numpy(wv))
            if fix_emb: 
               self.emb.weight.requires_grad = False

        self.rnn = nn.GRU(emb_size, h_size, num_layers=num_layer, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(h_size * 2, num_label)
        self.dropout = nn.Dropout(dropout)

        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(h_size * 2, h_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(h_size * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)


    def attention_net(self, x):       #x:[batch, seq_len, hidden_dim*2]

        u = torch.tanh(torch.matmul(x, self.w_omega))         #[batch, seq_len, hidden_dim*2]
        att = torch.matmul(u, self.u_omega)                   #[batch, seq_len, 1]
        att_score = torch.softmax(att, dim=1)

        scored_x = x * att_score                              #[batch, seq_len, hidden_dim*2]

        context = torch.sum(scored_x, dim=1)                  #[batch, hidden_dim*2]
        return context


    def forward(self, x):
        x = self.emb(x)       #[batch, seq_len, embedding_dim]
        x = self.dropout(x)
        # output: [batch, seq_len, hidden_dim*2]     hidden/cell: [n_layers*2, batch, hidden_dim]
        output, _ = self.rnn(x)

        attn_output = self.attention_net(output)
        attn_output = self.dropout(attn_output)
        logit = self.fc(attn_output)
        return torch.sigmoid(logit)


class TF_model(nn.Module):

    def __init__(self,
                emb_size,
                wv_model_file,
                num_label,    
                use_pretrain,
                fix_emb,
                nhead=2,
                dim_feedforward=128,
                dropout=0.1,
                maxlen=100,
                ):
        super().__init__()
        wv = read_pkl(wv_model_file).wv.vectors
        self.emb = nn.Embedding(num_embeddings=wv.shape[0]+1, embedding_dim=emb_size, padding_idx=0)
        if use_pretrain: 
            wv = np.concatenate( ( np.zeros((1, wv.shape[1])), wv ) )
            self.emb.weight.data.copy_(torch.from_numpy(wv))
            if fix_emb: 
               self.emb.weight.requires_grad = False

        self.pos_emb = nn.Embedding(maxlen, emb_size)

        self.tfe = TFE(emb_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.fc = nn.Linear(emb_size, num_label)
        self.maxlen = maxlen
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        x = self.emb(x)
        x = self.dp(x)
        pos = torch.tensor( range(self.maxlen), dtype=torch.long, device=next(self.parameters()).device )
        x += self.pos_emb(pos).unsqueeze(0)
        x = self.tfe(x).mean(dim=1)
        x = self.dp(x)
        return torch.sigmoid(self.fc(x))


class NN_model(nn.Module):

    def __init__(self,
                    input_dim,
                    num_label,
                    dp=0.,
                ):
        super().__init__()
        self.m = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(512, num_label),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.m(x)
        

class GRU_TF(nn.Module):

    def __init__(self,
                emb_size,
                wv_model_file,
                h_size,
                num_label,
                fix_emb=True,
                use_pretrain=True,
                bidirectional=False,
                num_layer=1,
                nhead=1,
                dim_feedforward=256,
                dropout=0.1,
              ):
        super().__init__()
        wv = read_pkl(wv_model_file).wv.vectors
        self.emb = nn.Embedding(num_embeddings=wv.shape[0]+1, embedding_dim=emb_size, padding_idx=0)
        if use_pretrain: 
            wv = np.concatenate( ( np.zeros((1, wv.shape[1])), wv ) )
            self.emb.weight.data.copy_(torch.from_numpy(wv))
            if fix_emb: 
               self.emb.weight.requires_grad = False
        self.gru = nn.GRU(emb_size, hidden_size=h_size, batch_first=True, dropout=dropout, bidirectional=bidirectional, num_layers=num_layer)
        tf_in_size = h_size * 2 if bidirectional else h_size 
        self.tfe = TFE(tf_in_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(tf_in_size, num_label)
  
    def forward(self, x):
        batch_size = x.size(0)
        x = self.emb(x)
        x, _ = self.gru(x) 
        x = self.tfe(x).mean(dim=1)
        x = self.dp(x)
        x = torch.sigmoid(self.fc(x))
        return x


class NN_GRU(nn.Module):

    def __init__(self,
                nn_input_dim,
                num_label,
                emb_size,
                wv_model_file,
                nn_h_size=512,
                h_size=256,
                fix_emb=True,
                use_pretrain=True,
                bidirectional=False,
                num_layer=1,
                dropout=0.,
                ):
        super().__init__()
        self.nn_model = nn.Sequential(
            nn.Linear(nn_input_dim, nn_h_size),
            nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nn_h_size, nn_h_size//2),
            nn.ReLU(),
        )
        wv = read_pkl(wv_model_file).wv.vectors
        self.emb = nn.Embedding(num_embeddings=wv.shape[0]+1, embedding_dim=emb_size, padding_idx=0)
        if use_pretrain: 
            wv = np.concatenate( ( np.zeros((1, wv.shape[1])), wv ) )
            self.emb.weight.data.copy_(torch.from_numpy(wv))
            if fix_emb: 
               self.emb.weight.requires_grad = False
        self.gru = nn.GRU(emb_size, hidden_size=h_size, batch_first=True, dropout=dropout, bidirectional=bidirectional, num_layers=num_layer)
        gru_out_dim = h_size*2*num_layer if bidirectional else h_size*num_layer
        self.fc = nn.Linear(gru_out_dim+nn_h_size//2, num_label)
        self.dp = nn.Dropout(dropout)
        
    def forward(self, x1, x2):
        x1 = self.nn_model(x1)
        batch_size = x2.size(0)
        x2 = self.emb(x2)
        _, x2 = self.gru(x2)
        x2 = x2.permute(1,0,2).reshape(batch_size, -1)
        x = torch.cat((x1, x2), dim=-1)
        x = self.dp(x)
        x = torch.sigmoid(self.fc(x))
        return x


class TextCNN(nn.Module):

    def __init__(self,
                emb_size, 
                num_label,
                wv_model_file,
                use_pretrain,
                fix_emb,
                kernel_num=100,
                kernel_size=[3,4,5],
                dropout=0.,
                ):
        super().__init__()
        wv = read_pkl(wv_model_file).wv.vectors
        self.emb = nn.Embedding(num_embeddings=wv.shape[0]+1, embedding_dim=emb_size, padding_idx=0)
        if use_pretrain: 
            wv = np.concatenate( ( np.zeros((1, wv.shape[1])), wv ) )
            self.emb.weight.data.copy_(torch.from_numpy(wv))
            if fix_emb: 
                self.emb.weight.requires_grad = False
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, emb_size)) for K in kernel_size])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_size) * kernel_num, num_label)

    def forward(self, x):
        x = self.emb(x)  # (N, W, D)-batch,单词数量，维度
        x = self.dropout(x)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        x = torch.sigmoid(self.fc(x))  # (N, C)
        return x


class TextCNN2(nn.Module):
    def __init__(self, 
                emb_size,
                wv_model_file,
                num_channels=100,
                kernel_size = [3,4,5],
                seq_max_len=100,
                num_label=17,
                fix_emb=True,
                dropout=0.,
                use_pretrain=True,
                ):
        super().__init__()

        wv = read_pkl(wv_model_file).wv.vectors
        self.emb = nn.Embedding(num_embeddings=wv.shape[0]+1, embedding_dim=emb_size, padding_idx=0)
        if use_pretrain: 
            wv = np.concatenate( ( np.zeros((1, wv.shape[1])), wv ) )
            self.emb.weight.data.copy_(torch.from_numpy(wv))
            if fix_emb: 
               self.emb.weight.requires_grad = False
        
        # Embedding Layer
        # self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        # self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        
        # This stackoverflow thread clarifies how conv1d works
        # https://stackoverflow.com/questions/46503816/keras-conv1d-layer-parameters-filters-and-kernel-size/46504997
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=emb_size, out_channels=num_channels, kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(seq_max_len - kernel_size[0]+1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(emb_size, out_channels=num_channels, kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(seq_max_len - kernel_size[1]+1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=emb_size, out_channels=num_channels, kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(seq_max_len - kernel_size[2]+1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Fully-Connected Layer
        self.fc = nn.Linear(num_channels*len(kernel_size), num_label)
        
    def forward(self, x):
        # x.shape = (max_sen_len, batch_size)
        embedded_sent = self.emb(x).permute(0,2,1)
        embedded_sent = self.dropout(embedded_sent)
        # embedded_sent.shape = (batch_size=64,embed_size=300,max_sen_len=20)
        
        conv_out1 = self.conv1(embedded_sent).squeeze(2) #shape=(64, num_channels, 1) (squeeze 1)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)
        
        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        final_out = self.fc(final_feature_map)
        return torch.sigmoid(final_out)


class RCNN(nn.Module):
    def __init__(self, 
                emb_size,
                wv_model_file,
                h_size,
                num_label,
                fix_emb=True,
                dropout=0.,
                use_pretrain=True,
                num_layer=1,
        ):
        super().__init__()
        
        wv = read_pkl(wv_model_file).wv.vectors
        self.emb = nn.Embedding(num_embeddings=wv.shape[0]+1, embedding_dim=emb_size, padding_idx=0)
        if use_pretrain: 
            wv = np.concatenate( ( np.zeros((1, wv.shape[1])), wv ) )
            self.emb.weight.data.copy_(torch.from_numpy(wv))
            if fix_emb: 
               self.emb.weight.requires_grad = False
        
        # Bi-directional LSTM for RCNN
        self.lstm = nn.LSTM(emb_size,
                            h_size,
                            num_layers=num_layer,
                            dropout=dropout,
                            bidirectional=True,
                            batch_first=True,
                            )
        
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer to get "convolution output" to be passed to Pooling Layer
        self.W = nn.Linear(
            emb_size + 2*h_size,
            h_size,
        )
        
        # Tanh non-linearity
        self.tanh = nn.Tanh()
        
        # Fully-Connected Layer
        self.fc = nn.Linear(
            h_size,
            num_label,
        )
        
    def forward(self, x):
        # x.shape = (seq_len, batch_size)
        embedded_sent = self.emb(x)
        # embedded_sent.shape = (seq_len, batch_size, embed_size)
        embedded_sent = self.dropout(embedded_sent)   # embedding dropout

        lstm_out, (h_n,c_n) = self.lstm(embedded_sent)
        # lstm_out.shape = (seq_len, batch_size, 2 * hidden_size)
        
        input_features = torch.cat([lstm_out,embedded_sent], 2)   # .permute(1,0,2)
        # final_features.shape = (batch_size, seq_len, embed_size + 2*hidden_size)
        
        linear_output = self.tanh(
            self.W(input_features)
        )
        # linear_output.shape = (batch_size, seq_len, hidden_size_linear)
        
        linear_output = linear_output.permute(0,2,1) # Reshaping fot max_pool
        
        max_out_features = nn.functional.max_pool1d(linear_output, linear_output.shape[2]).squeeze(2)
        # max_out_features.shape = (batch_size, hidden_size_linear)
        
        max_out_features = self.dropout(max_out_features)
        final_out = self.fc(max_out_features)
        return torch.sigmoid(final_out)


class RCNN2(nn.Module):
    def __init__(self, 
                emb_size,
                wv_model_file,
                h_size,
                num_label,
                fix_emb=True,
                dropout=0.,
                use_pretrain=True,
                num_layer=1,
        ):
        super().__init__()
        
        wv = read_pkl(wv_model_file).wv.vectors
        self.emb = nn.Embedding(num_embeddings=wv.shape[0]+1, embedding_dim=emb_size, padding_idx=0)
        if use_pretrain: 
            wv = np.concatenate( ( np.zeros((1, wv.shape[1])), wv ) )
            self.emb.weight.data.copy_(torch.from_numpy(wv))
            if fix_emb: 
               self.emb.weight.requires_grad = False
        
        # Bi-directional LSTM for RCNN
        self.gru = nn.GRU(emb_size,
                            h_size,
                            num_layers=num_layer,
                            dropout=dropout,
                            bidirectional=True,
                            batch_first=True,
                            )
        
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer to get "convolution output" to be passed to Pooling Layer
        self.W = nn.Linear(
            emb_size + 2*h_size,
            h_size,
        )
        
        # Tanh non-linearity
        self.tanh = nn.Tanh()
        
        # Fully-Connected Layer
        self.fc = nn.Linear(
            h_size,
            num_label,
        )
        
    def forward(self, x):
        # x.shape = (seq_len, batch_size)
        embedded_sent = self.emb(x)
        embedded_sent = self.dropout(embedded_sent)
        # embedded_sent.shape = (seq_len, batch_size, embed_size)

        gru_out, _ = self.gru(embedded_sent)
        # lstm_out.shape = (seq_len, batch_size, 2 * hidden_size)
        
        input_features = torch.cat([gru_out,embedded_sent], 2)   # .permute(1,0,2)
        # final_features.shape = (batch_size, seq_len, embed_size + 2*hidden_size)
        
        linear_output = self.tanh(
            self.W(input_features)
        )
        # linear_output.shape = (batch_size, seq_len, hidden_size_linear)
        
        linear_output = linear_output.permute(0,2,1) # Reshaping fot max_pool
        
        max_out_features = nn.functional.max_pool1d(linear_output, linear_output.shape[2]).squeeze(2)
        # max_out_features.shape = (batch_size, hidden_size_linear)
        
        max_out_features = self.dropout(max_out_features)
        final_out = self.fc(max_out_features)
        return torch.sigmoid(final_out)



class Seq2SeqAtten(nn.Module):
    def __init__(self, 
                emb_size,
                wv_model_file,
                h_size,
                num_label,
                fix_emb=True,
                dropout=0.,
                use_pretrain=True,
                bidirectional=True,
                num_layer=1,
                ):
        super().__init__()
        # self.config = config

        wv = read_pkl(wv_model_file).wv.vectors
        self.emb = nn.Embedding(num_embeddings=wv.shape[0]+1, embedding_dim=emb_size, padding_idx=0)
        if use_pretrain: 
            wv = np.concatenate( ( np.zeros((1, wv.shape[1])), wv ) )
            self.emb.weight.data.copy_(torch.from_numpy(wv))
            if fix_emb: 
               self.emb.weight.requires_grad = False
        
        # Embedding Layer
        # self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        # self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        
        # Encoder RNN
        self.lstm = nn.LSTM(input_size = emb_size,
                            hidden_size = h_size,
                            num_layers = num_layer,
                            bidirectional = bidirectional,
                            batch_first=True
                            )
        
        # Dropout Layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully-Connected Layer
        self.fc = nn.Linear(
            h_size * (1+bidirectional) * 2,
            num_label
        )

        self.num_layer = num_layer
        self.bid = bidirectional
        self.h_size = h_size
        
        # Softmax non-linearity
        # self.softmax = nn.Softmax()
                
    def apply_attention(self, rnn_output, final_hidden_state):
        '''
        Apply Attention on RNN output
        
        Input:
            rnn_output (batch_size, seq_len, num_directions * hidden_size): tensor representing hidden state for every word in the sentence
            final_hidden_state (batch_size, num_directions * hidden_size): final hidden state of the RNN
            
        Returns:
            attention_output(batch_size, num_directions * hidden_size): attention output vector for the batch
        '''
        hidden_state = final_hidden_state.unsqueeze(2)
        attention_scores = torch.bmm(rnn_output, hidden_state).squeeze(2)
        soft_attention_weights = torch.softmax(attention_scores, 1).unsqueeze(2) #shape = (batch_size, seq_len, 1)
        attention_output = torch.bmm(rnn_output.permute(0,2,1), soft_attention_weights).squeeze(2)
        return attention_output
        
    def forward(self, x):
        # x.shape = (max_sen_len, batch_size)
        batch_size = x.shape[0]
        embedded_sent = self.emb(x)
        embedded_sent = self.dropout(embedded_sent)
        # embedded_sent.shape = (max_sen_len=20, batch_size=64,embed_size=300)

        ##################################### Encoder #######################################
        lstm_output, (h_n,c_n) = self.lstm(embedded_sent)
        # lstm_output.shape = (seq_len, batch_size, num_directions * hidden_size)
        
        # Final hidden state of last layer (num_directions, batch_size, hidden_size)
        h_n_final_layer = h_n.view(self.num_layer,
                                   self.bid + 1,
                                   batch_size,
                                   self.h_size)[-1,:,:,:]
        
        ##################################### Attention #####################################
        # Convert input to (batch_size, num_directions * hidden_size) for attention
        final_hidden_state = torch.cat([h_n_final_layer[i,:,:] for i in range(h_n_final_layer.shape[0])], dim=1)
        
        attention_out = self.apply_attention(lstm_output, final_hidden_state)
        # Attention_out.shape = (batch_size, num_directions * hidden_size)
        
        #################################### Linear #########################################
        concatenated_vector = torch.cat([final_hidden_state, attention_out], dim=1)
        final_feature_map = self.dropout(concatenated_vector) # shape=(batch_size, num_directions * hidden_size)
        final_out = self.fc(final_feature_map)
        return torch.sigmoid(final_out)


class Seq2SeqAtten2(nn.Module):
    def __init__(self, 
                emb_size,
                wv_model_file,
                h_size,
                num_label,
                fix_emb=True,
                dropout=0.,
                use_pretrain=True,
                bidirectional=True,
                num_layer=1,
                ):
        super().__init__()
        # self.config = config

        wv = read_pkl(wv_model_file).wv.vectors
        self.emb = nn.Embedding(num_embeddings=wv.shape[0]+1, embedding_dim=emb_size, padding_idx=0)
        if use_pretrain: 
            wv = np.concatenate( ( np.zeros((1, wv.shape[1])), wv ) )
            self.emb.weight.data.copy_(torch.from_numpy(wv))
            if fix_emb: 
               self.emb.weight.requires_grad = False
        
        # Embedding Layer
        # self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        # self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        
        # Encoder RNN
        self.gru = nn.GRU(input_size = emb_size,
                            hidden_size = h_size,
                            num_layers = num_layer,
                            bidirectional = bidirectional,
                            batch_first=True
                            )
        
        # Dropout Layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully-Connected Layer
        self.fc = nn.Linear(
            h_size * (1+bidirectional) * 2,
            num_label
        )

        self.num_layer = num_layer
        self.bid = bidirectional
        self.h_size = h_size
        
        # Softmax non-linearity
        # self.softmax = nn.Softmax()
                
    def apply_attention(self, rnn_output, final_hidden_state):
        '''
        Apply Attention on RNN output
        
        Input:
            rnn_output (batch_size, seq_len, num_directions * hidden_size): tensor representing hidden state for every word in the sentence
            final_hidden_state (batch_size, num_directions * hidden_size): final hidden state of the RNN
            
        Returns:
            attention_output(batch_size, num_directions * hidden_size): attention output vector for the batch
        '''
        hidden_state = final_hidden_state.unsqueeze(2)
        attention_scores = torch.bmm(rnn_output, hidden_state).squeeze(2)
        soft_attention_weights = torch.softmax(attention_scores, 1).unsqueeze(2) #shape = (batch_size, seq_len, 1)
        attention_output = torch.bmm(rnn_output.permute(0,2,1), soft_attention_weights).squeeze(2)
        return attention_output
        
    def forward(self, x):
        # x.shape = (max_sen_len, batch_size)
        batch_size = x.shape[0]
        embedded_sent = self.emb(x)
        # embedded_sent.shape = (max_sen_len=20, batch_size=64,embed_size=300)
        embedded_sent = self.dropout(embedded_sent)

        ##################################### Encoder #######################################
        gru_output, h_n = self.gru(embedded_sent)
        # lstm_output.shape = (seq_len, batch_size, num_directions * hidden_size)
        
        # Final hidden state of last layer (num_directions, batch_size, hidden_size)
        h_n_final_layer = h_n.view(self.num_layer,
                                   self.bid + 1,
                                   batch_size,
                                   self.h_size)[-1,:,:,:]
        
        ##################################### Attention #####################################
        # Convert input to (batch_size, num_directions * hidden_size) for attention
        final_hidden_state = torch.cat([h_n_final_layer[i,:,:] for i in range(h_n_final_layer.shape[0])], dim=1)
        
        attention_out = self.apply_attention(gru_output, final_hidden_state)
        # Attention_out.shape = (batch_size, num_directions * hidden_size)
        
        #################################### Linear #########################################
        concatenated_vector = torch.cat([final_hidden_state, attention_out], dim=1)
        final_feature_map = self.dropout(concatenated_vector) # shape=(batch_size, num_directions * hidden_size)
        final_out = self.fc(final_feature_map)
        return torch.sigmoid(final_out)


class FastText(nn.Module):
    def __init__(self, 
                emb_size,
                wv_model_file,
                h_size,
                num_label,    
                use_pretrain,
                fix_emb,
                dropout=0,
                    ):
        super().__init__()
        wv = read_pkl(wv_model_file).wv.vectors
        self.emb = nn.Embedding(num_embeddings=wv.shape[0]+1, embedding_dim=emb_size, padding_idx=0)
        if use_pretrain:
            assert emb_size == wv.shape[1]
            wv = np.concatenate( ( np.zeros((1, wv.shape[1])), wv ) )
            self.emb.weight.data.copy_(torch.from_numpy(wv))
            if fix_emb: 
               self.emb.weight.requires_grad = False
        
        # Hidden Layer
        self.fc1 = nn.Linear(emb_size, h_size)
        
        # Output Layer
        self.fc2 = nn.Linear(h_size, num_label)

        self.dp = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.emb(x)
        h = self.fc1(x.mean(1))
        h = self.dp(h)
        z = self.fc2(h)
        return torch.sigmoid(z)


class CNN_BiGRU(nn.Module):
    
    def __init__(self,
                 wv_model_file,
                 emb_size=128,
                 h_size=256,
                 k_size=[3,3,3],
                 num_layer=1,
                 num_label=17,
                 dropout=.25,
                 use_pretrain=False,
                 fix_emb=False,
                ):
        super().__init__()
        wv = read_pkl(wv_model_file).wv.vectors
        self.emb = nn.Embedding(num_embeddings=wv.shape[0]+1, embedding_dim=emb_size, padding_idx=0)
        if use_pretrain: 
            wv = np.concatenate( ( np.zeros((1, wv.shape[1])), wv ) )
            self.emb.weight.data.copy_(torch.from_numpy(wv))
            if fix_emb: 
                self.emb.weight.requires_grad = False
        self.conv1 = nn.Conv1d(emb_size, emb_size, k_size[0])
        self.conv2 = nn.Conv1d(emb_size, emb_size,  k_size[1])
        self.conv3 = nn.Conv1d(emb_size, emb_size,  k_size[2])
        self.dp = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_size, h_size, bidirectional=True, batch_first=True, num_layers=num_layer)
        self.fc = nn.Linear(2*h_size, num_label)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.emb(x)
        x = self.dp(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        _, x = self.rnn(x.permute(0,2,1))
        x = self.dp(x.permute(1,0,2).contiguous().view(batch_size, -1))
        x = torch.sigmoid(self.fc(x))
        return x


# def crt_model(F):
#     if F.model_name.lower() == "baseline":
#         return Baseline(
#             F.emb_size, 
#             F.wv_model_file,
#             F.rnn_dim,
#             F.num_label,
#             fix_emb=F.fix_emb,
#             dropout=F.dropout,
#             use_pretrain=F.emb_pretrain,
#         )
#     elif F.model_name.lower() == "lstm":
#         return LSTM_model(
#             F.emb_size, 
#             F.wv_model_file,
#             F.rnn_dim,
#             F.num_label,
#             fix_emb=F.fix_emb,
#             dropout=F.dropout,
#             use_pretrain=F.emb_pretrain,
#             bidirectional=F.rnn_bid,
#             num_layer=F.rnn_num_layer,
#         )
#     elif F.model_name.lower() == "gru":
#         return GRU_model(
#             F.emb_size, 
#             F.wv_model_file,
#             F.rnn_dim,
#             F.num_label,
#             fix_emb=F.fix_emb,
#             dropout=F.dropout,
#             use_pretrain=F.emb_pretrain,
#             bidirectional=F.rnn_bid,
#             num_layer=F.rnn_num_layer,
#         )
#     elif F.model_name.lower() == "transformer":
#         return TF_model(
#             F.emb_size,
#             F.wv_model_file,
#             F.num_label,
#             F.emb_pretrain,
#             F.fix_emb,
#             dim_feedforward=128,
#             dropout=F.dropout,
#             nhead=F.tfe_nhead,
#         )
#     elif F.model_name.lower() == "nn":
#         return NN_model(
#             F.feat_dim,
#             F.num_label,
#             F.dropout,
#         )
#     elif F.model_name.lower() == "gru_tf":
#         return GRU_TF(
#                 emb_size=F.emb_size,
#                 wv_model_file=F.wv_model_file,
#                 h_size=F.rnn_dim,
#                 num_label=F.num_label,
#                 fix_emb=F.fix_emb,
#                 use_pretrain=F.emb_pretrain,
#                 bidirectional=F.rnn_bid,
#                 num_layer=F.rnn_num_layer,
#                 nhead=F.tfe_nhead,
#                 dim_feedforward=256,
#                 dropout=F.dropout,
#               )
#     elif F.model_name.lower() == "nn_gru":
#         return NN_GRU(
#                 nn_input_dim=F.feat_dim,
#                 num_label=F.num_label,
#                 emb_size=F.emb_size,
#                 wv_model_file=F.wv_model_file,
#                 nn_h_size=512,
#                 h_size=F.rnn_dim,
#                 fix_emb=F.fix_emb,
#                 use_pretrain=F.emb_pretrain,
#                 bidirectional=F.rnn_bid,
#                 num_layer=F.rnn_num_layer,
#                 dropout=F.dropout,
#         )
#     elif F.model_name.lower() == "textcnn":
#         return TextCNN(
#             emb_size=F.emb_size, 
#             num_label=F.num_label,
#             wv_model_file=F.wv_model_file,
#             use_pretrain=F.emb_pretrain,
#             fix_emb=F.fix_emb,
#             kernel_num=100,
#             kernel_size=[3,4,5],
#             dropout=F.dropout,
#         )
#     elif F.model_name.lower() == "bilstm_atten":
#         return BiLSTM_Atten(
#             F.emb_size,
#             F.wv_model_file,
#             F.rnn_dim,
#             F.num_label,
#             F.fix_emb,
#             F.dropout,
#             F.emb_pretrain,
#             F.rnn_num_layer,
#         )
#     elif F.model_name.lower() == "bilstm_atten2":
#         return BiLSTM_Atten2(
#             F.emb_size,
#             F.wv_model_file,
#             F.rnn_dim,
#             F.num_label,
#             F.fix_emb,
#             F.dropout,
#             F.emb_pretrain,
#             F.rnn_num_layer,
#         )

#     elif F.model_name.lower() == "maskgru_model":
#         return MaskGRU_model(
#                 emb_size=F.emb_size,
#                 wv_model_file=F.wv_model_file,
#                 h_size=F.rnn_dim,
#                 num_label=F.num_label,
#                 fix_emb=F.fix_emb,
#                 dropout=F.dropout,
#                 use_pretrain=F.emb_pretrain,
#                 bidirectional=F.rnn_bid,
#                 num_layer=F.rnn_num_layer,
#         )
#     elif F.model_name.lower() == "gru_model2":
#         return GRU_model2(
#             F.emb_size, 
#             F.wv_model_file,
#             F.rnn_dim,
#             F.num_label,
#             fix_emb=F.fix_emb,
#             dropout=F.dropout,
#             use_pretrain=F.emb_pretrain,
#             bidirectional=F.rnn_bid,
#             num_layer=F.rnn_num_layer,
#         )
#     elif F.model_name.lower() == "bigru_atten2":
#         return BiGRU_Atten2(
#             F.emb_size,
#             F.wv_model_file,
#             F.rnn_dim,
#             F.num_label,
#             F.fix_emb,
#             F.dropout,
#             F.emb_pretrain,
#             F.rnn_num_layer,
#         )
#     elif F.model_name.lower() == "rcnn":
#         return RCNN(
#             F.emb_size,
#             F.wv_model_file,
#             F.rnn_dim,
#             F.num_label,
#             F.fix_emb,
#             F.dropout,
#             F.emb_pretrain,
#             F.rnn_num_layer,
#         )
#     elif F.model_name.lower() == "rcnn2":
#         return RCNN2(
#             F.emb_size,
#             F.wv_model_file,
#             F.rnn_dim,
#             F.num_label,
#             F.fix_emb,
#             F.dropout,
#             F.emb_pretrain,
#             F.rnn_num_layer,
#         )
#     elif F.model_name.lower() == "bigru_atten":
#         return BiGRU_Atten(
#             F.emb_size,
#             F.wv_model_file,
#             F.rnn_dim,
#             F.num_label,
#             F.fix_emb,
#             F.dropout,
#             F.emb_pretrain,
#             F.rnn_num_layer,
#         )
#     elif F.model_name.lower() == "textcnn2":
#         return TextCNN2(
#                 emb_size=F.emb_size,
#                 wv_model_file=F.wv_model_file,
#                 num_channels=100,
#                 kernel_size = [3,4,5],
#                 seq_max_len=F.desc_max_len,
#                 num_label=F.num_label,
#                 fix_emb=F.fix_emb,
#                 dropout=F.dropout,
#                 use_pretrain=F.emb_pretrain,
#         )
#     elif F.model_name.lower() == "seq2seqatten":
#         return Seq2SeqAtten(
#             F.emb_size,
#             F.wv_model_file,
#             F.rnn_dim,
#             F.num_label,
#             F.fix_emb,
#             F.dropout,
#             F.emb_pretrain,
#             F.rnn_bid,
#             F.rnn_num_layer,
#         )
#     elif F.model_name.lower() == "seq2seqatten2":
#         return Seq2SeqAtten2(
#             F.emb_size,
#             F.wv_model_file,
#             F.rnn_dim,
#             F.num_label,
#             F.fix_emb,
#             F.dropout,
#             F.emb_pretrain,
#             F.rnn_bid,
#             F.rnn_num_layer,
#         )
#     elif F.model_name.lower() == "fasttext":
#         return FastText(
#                 emb_size=F.emb_size,
#                 wv_model_file=F.wv_model_file,
#                 h_size=32,
#                 num_label=F.num_label,    
#                 use_pretrain=F.emb_pretrain,
#                 fix_emb=F.fix_emb,
#                 dropout=F.dropout,
#             )
#     elif F.model_name.lower() == "cnn_bigru":
#         return CNN_BiGRU(
#             F.wv_model_file,
#             F.emb_size,
#             F.rnn_dim,
#             num_label=F.num_label,
#             dropout=F.dropout,
#             num_layer=F.rnn_num_layer,
#             use_pretrain=F.emb_pretrain,
#             fix_emb=F.fix_emb,
#         )

#     else:
#         print("model not found!")
#         sys.exit(-1)


# if __name__ == "__main__":

    # model = Baseline(
    #     128, 
    #     'data/wv_model_128_10_10_5.pkl',
    #     256,
    #     17,
    # )

    # x = torch.rand(5, 90).long()
    # y = model(x)
    # print(y.shape)
    
    # m = MaskGRU(5, 10)
    # x = torch.rand(2, 3, 5)
    # y = m(x, bid=True, mask_zero=True)

