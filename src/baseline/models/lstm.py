# -*- coding: utf-8 -*-


import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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


class LSTMModule(nn.Module):
    def __init__(self, tag=True):
        super(LSTMModule, self).__init__()
        indim = 32 * 4 * 2
        if tag:
            indim += 7
        self.lstm = nn.LSTM(input_size=indim, hidden_size=128, num_layers=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        context = x[:6, :, :]
        queries = [x[i:i + 1, :, :] for i in range(6, 10)]
        _, (h_n, c_n) = self.lstm(context)
        hidden = [self.lstm(query, (h_n, c_n))[0].permute(1, 0, 2) for query in queries]
        hidden = torch.cat(hidden, dim=1)
        pred = self.fc(hidden.view(-1, 128)).view(-1, 4, 3)
        return pred


class LSTM(nn.Module):
    def __init__(self, tag=True):
        super(LSTM, self).__init__()
        self.use_tag = tag
        self.conv = ConvModule()
        self.lstm = LSTMModule(self.use_tag)
        if self.use_tag:
            self.register_buffer("tags", torch.tensor(self.build_tags(), dtype=torch.float))

    def build_tags(self):
        tags = np.zeros((10, 7))
        tags[:6, :6] = np.eye(6)
        tags[6:, -1] = 1
        return tags

    def forward(self, x):
        batch = x.shape[0]
        features = self.conv(x.view(-1, 3, 60, 80)).view(-1, 10, 256)
        if self.use_tag:
            features = torch.cat([features, self.tags.unsqueeze(0).expand(batch, -1, -1)], dim=2)
        pred = self.lstm(features)
        return pred