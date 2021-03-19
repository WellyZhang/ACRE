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
        self.conv1 = nn.Conv2d(21, 32, kernel_size=3, stride=2, bias=False)
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


class MLPModule(nn.Module):
    def __init__(self):
        super(MLPModule, self).__init__()
        self.fc1 = nn.Linear(32 * 2 * 4, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 3)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = ConvModule()
        self.mlp = MLPModule()

    def forward(self, x):
        context = x[:, :6, :, :, :]
        query = x[:, 6:, :, :, :]

        context = context.unsqueeze(1).expand(-1, 4, -1, -1, -1, -1)
        query = query.unsqueeze(2)

        x_in = torch.cat((context, query), dim=2).view(-1, 21, 60, 80)
        features = self.conv(x_in)
        pred = self.mlp(features).view(-1, 4, 3)
        return pred