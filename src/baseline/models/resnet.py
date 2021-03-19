# -*- coding: utf-8 -*-


import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


class MLPModule(nn.Module):
    def __init__(self):
        super(MLPModule, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 3)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(21, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = Identity()
        self.mlp = MLPModule()

    def forward(self, x):
        context = x[:, :6, :, :, :]
        query = x[:, 6:, :, :, :]

        context = context.unsqueeze(1).expand(-1, 4, -1, -1, -1, -1)
        query = query.unsqueeze(2)

        x_in = torch.cat((context, query), dim=2).view(-1, 21, 224, 224)
        features = self.resnet18(x_in)
        pred = self.mlp(features).view(-1, 4, 3)
        return pred

    