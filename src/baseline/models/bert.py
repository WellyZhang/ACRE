# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ConvModule(nn.Module):
    def __init__(self):
        super(ConvModule, self).__init__()
        # set bias=False before BatchNorm
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, bias=False)
        self.batch_norm4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.batch_norm1(x))
        x = self.conv2(x)
        x = self.relu2(self.batch_norm2(x))
        x = self.conv3(x)
        x = self.relu3(self.batch_norm3(x))
        x = self.conv4(x)
        x = self.relu4(self.batch_norm4(x))
        return x.view(-1, 32 * 2 * 4)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :, :]
        return self.dropout(x)


class BERT(nn.Module):
    def __init__(self, ninp=256, nhead=8, nhid=1024, nlayers=12, dropout=0.1):
        super(BERT, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.special_embeddings = nn.Embedding(2, ninp)
        self.segment_embeddings = nn.Embedding(2, ninp)
        self.encoder = ConvModule()
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, 3)

    def forward(self, src):
        src = self.encoder(src.view(-1, 3, 60, 80)).view(-1, 10, 256)
        context = src[:, :6, :].unsqueeze(1).expand(-1, 4, -1, -1)
        choice = src[:, 6:, :].unsqueeze(2)

        cls_symbol = self.special_embeddings(torch.zeros(choice.shape[:-1], dtype=torch.long, device=src.device))
        sep_symbol = self.special_embeddings(torch.ones(choice.shape[:-1], dtype=torch.long, device=src.device))

        src = torch.cat((cls_symbol, context, sep_symbol, choice, sep_symbol), dim=2).view(-1, 10, 256)
        src = src.permute(1, 0, 2)

        question_segment = self.segment_embeddings(torch.zeros((8, 1), dtype=torch.long, device=src.device))
        answer_segment = self.segment_embeddings(torch.ones((2, 1), dtype=torch.long, device=src.device))
        se = torch.cat((question_segment, answer_segment), dim=0)

        src = src * np.sqrt(self.ninp)
        src = self.pos_encoder(src + se)
        output = self.transformer_encoder(src)
        cls_features = output[0, :, :]
        output = self.decoder(cls_features).view(-1, 4, 3)
        return output