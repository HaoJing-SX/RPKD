import torch.nn as nn
import pdb
import torch
import numpy as np
from numpy import *
import torch.nn.functional as F
from typing import Optional, List
from torch import Tensor
import copy
from copy import deepcopy
from ...utils import common_utils, spconv_utils


class PositionalEmbedding(nn.Module):
    def __init__(self, demb=256):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    # pos_seq =  pos_seq = torch.arange(seq_len-1, -1, -1.0)
    def forward(self, pos_seq, batch_size=2):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if batch_size is not None:
            return pos_emb[:, None, :].expand(-1, batch_size, -1)
        else:
            return pos_emb[:, None, :]

class CrossAttention(nn.Module):

    def __init__(self, hidden_dim):
        super(CrossAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.pos_dim = 8

        self.pos_en = PositionalEmbedding(self.pos_dim)

        self.Q_linear = nn.Linear(hidden_dim+self.pos_dim, hidden_dim, bias=False)
        self.K_linear = nn.Linear(hidden_dim+self.pos_dim, hidden_dim, bias=False)
        self.V_linear = nn.Linear(hidden_dim+self.pos_dim, hidden_dim, bias=False)

        self.att = nn.MultiheadAttention(hidden_dim, 4)


    def forward(self, inputs, Q_in): # N,B,C

        batch_size = inputs.shape[1]
        seq_len = inputs.shape[0]

        pos_input = torch.from_numpy(np.arange(seq_len)+1).cuda()
        pos_input = self.pos_en(pos_input, batch_size)
        inputs_pos = torch.cat([inputs, pos_input], -1)

        pos_Q = torch.from_numpy(np.array([seq_len])).cuda()
        pos_Q = self.pos_en(pos_Q, batch_size)

        Q_in_pos = torch.cat([Q_in, pos_Q], -1)

        Q = self.Q_linear(Q_in_pos)
        K = self.K_linear(inputs_pos)
        V = self.V_linear(inputs_pos)

        out = self.att(Q, K, V)

        return out[0]

class EfficientAttention(nn.Module):

    def __init__(self, hidden_dim):
        super(EfficientAttention, self).__init__()

        self.hidden_dim = hidden_dim
        # self.cin = cin
        self.pos_dim = 8

        self.pos_en = PositionalEmbedding(self.pos_dim)

        self.Q_linear = nn.Linear(hidden_dim+self.pos_dim, hidden_dim, bias=False)
        self.K_linear = nn.Linear(hidden_dim+self.pos_dim, hidden_dim, bias=False)
        self.V_linear = nn.Linear(hidden_dim+self.pos_dim, hidden_dim, bias=False)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # self.bn_1 = nn.BatchNorm1d(640, eps=1e-3, momentum=0.01, affine=False, track_running_stats=True)
        # self.bn_2 = nn.BatchNorm1d(400, eps=1e-3, momentum=0.01, affine=False, track_running_stats=True)
        self.mlp = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=True)

        self.FFN = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, inputs, Q_in): # N,B,C

        batch_size = inputs.shape[1]
        seq_len = inputs.shape[0]

        pos_input = torch.from_numpy(np.arange(seq_len)+1).cuda()
        pos_input = self.pos_en(pos_input, batch_size)
        inputs_pos = torch.cat([inputs, pos_input], -1)

        pos_Q = torch.from_numpy(np.array([seq_len])).cuda()
        pos_Q = self.pos_en(pos_Q, batch_size)
        # print('Q_in: ', Q_in.size())
        Q_in_pos = torch.cat([Q_in, pos_Q], -1)

        Q = self.Q_linear(Q_in_pos)
        K = self.K_linear(inputs_pos)
        V = self.V_linear(inputs_pos)

        Q_1 = F.softmax(Q, dim=2)
        K_1 = F.softmax(K, dim=2).permute(0, 2, 1)

        alpha = torch.matmul(K_1, V)
        # print('Q_1, alpha: ', Q_1, alpha)
        bata = torch.matmul(Q_1, alpha)
        # print('bata: ', bata.size())

        out_1 = bata + inputs
        out_2 = self.FFN(out_1)
        out_3 = self.norm1(out_2)
        out = out_3 + out_1

        return out[0]