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


class RelationModule(nn.Module):
    def __init__(self):
        super(RelationModule, self).__init__()
        self.fc1 = nn.Linear(256 * 2, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 512)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(512, 256)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        return x


class MLPModule(nn.Module):
    def __init__(self):
        super(MLPModule, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class PanelsToEmbeddings(nn.Module):
    def __init__(self, tag):
        super(PanelsToEmbeddings, self).__init__()
        self.in_dim = 256
        if tag:
            self.in_dim += 7
        self.fc = nn.Linear(self.in_dim, 256)

    def forward(self, x):
        return self.fc(x.view(-1, self.in_dim))


class WReN(nn.Module):
    def __init__(self, tag=True):
        super(WReN, self).__init__()
        self.use_tag = tag
        self.conv = ConvModule()
        self.rn = RelationModule()
        self.mlp = MLPModule()
        self.proj = PanelsToEmbeddings(self.use_tag)
        if self.use_tag:
            self.register_buffer("tags", torch.tensor(self.build_tags(), dtype=torch.float))

    def build_tags(self):
        tags = np.zeros((10, 7))
        tags[:6, :6] = np.eye(6)
        tags[6:, -1] = 1
        return tags

    def group_panel_embeddings_batch(self, embeddings):
        embeddings = embeddings.view(-1, 10, 256)
        context_embeddings = embeddings[:, :6, :]
        choice_embeddings = embeddings[:, 6:, :]
        context_embeddings_pairs = torch.cat((context_embeddings.unsqueeze(1).expand(-1, 6, -1, -1), context_embeddings.unsqueeze(2).expand(-1, -1, 6, -1)), dim=3).view(-1, 36, 512)
        
        context_embeddings = context_embeddings.unsqueeze(1).expand(-1, 4, -1, -1)
        choice_embeddings = choice_embeddings.unsqueeze(2).expand(-1, -1, 6, -1)
        choice_context_order = torch.cat((context_embeddings, choice_embeddings), dim=3)
        choice_context_reverse = torch.cat((choice_embeddings, context_embeddings), dim=3)
        embedding_paris = [context_embeddings_pairs.unsqueeze(1).expand(-1, 4, -1, -1), choice_context_order, choice_context_reverse]
        return torch.cat(embedding_paris, dim=2).view(-1, 4, 48, 512)

    def rn_sum_features(self, features):
        features = features.view(-1, 4, 48, 256)
        sum_features = torch.sum(features, dim=2)
        return sum_features

    def forward(self, x):
        batch_size = x.shape[0]
        panel_features = self.conv(x.view(-1, 3, 60, 80)).view(-1, 10, 256)
        if self.use_tag:
            expanded_tags = self.tags.unsqueeze(0).expand(batch_size, -1, -1)
            panel_features = torch.cat((panel_features, expanded_tags), dim=2)
        panel_embeddings = self.proj(panel_features)
        panel_embeddings_pairs = self.group_panel_embeddings_batch(panel_embeddings)
        panel_embedding_features = self.rn(panel_embeddings_pairs.view(-1, 512))
        sum_features = self.rn_sum_features(panel_embedding_features)
        pred = self.mlp(sum_features.view(-1, 256)).view(-1, 4, 3)
        return pred
