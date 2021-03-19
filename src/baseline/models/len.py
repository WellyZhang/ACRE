# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F

from torch import nn

class LEN(nn.Module):
    def __init__(self):
        super(LEN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pre_g_fc = nn.Linear(32 * 2 * 4, 256, bias=False)
        self.pre_g_batch_norm = nn.BatchNorm1d(256)

        self.cnn_global = nn.Sequential(
            nn.Conv2d(18, 32, 3, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pre_g_fc2 = nn.Linear(32 * 2 * 4, 256, bias=False)
        self.pre_g_batch_norm2 = nn.BatchNorm1d(256)

        self.g = nn.Sequential(
            nn.Linear(768, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * 3, bias=False),
            nn.BatchNorm1d(256 * 3),
            nn.ReLU(),
            nn.Linear(256 * 3, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout()
        )

        self.f = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 3)
        )

    def deeper_features(self, context_pairs, global_features, size, keepdim=False):
        out = torch.cat((context_pairs, global_features.unsqueeze(1).expand(-1, size, -1)), dim=2)
        out = out.view(-1, 768)
        out = self.g(out)
        out = out.view(-1, size, 512)
        out = torch.sum(out, dim=1, keepdim=keepdim)
        return out
    
    def context_combination(self, embeddings):
        embeddings = embeddings.view(-1, 6, 256)
        embeddings_pairs = torch.cat((embeddings.unsqueeze(1).expand(-1, 6, -1, -1), embeddings.unsqueeze(2).expand(-1, -1, 6, -1)), dim=3).view(-1, 36, 512)
        return embeddings_pairs

    def context_choice_combination(self, embeddings):
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

    def forward(self, x):
        # panel embeddings
        panel_embeddings = self.cnn(x.view(-1, 3, 60, 80))
        panel_embeddings = panel_embeddings.view(-1, 256)
        panel_embeddings = self.pre_g_fc(panel_embeddings)
        panel_embeddings = self.pre_g_batch_norm(panel_embeddings)
        panel_embeddings = F.relu(panel_embeddings)
        panel_embeddings = panel_embeddings.view(-1, 10, 256)

        context_embeddings = panel_embeddings[:, :6, :] 
        answer_embeddings = panel_embeddings[:, 6:, :]

        # global features
        context = x[:, :6, :, :, :].view(-1, 18, 60, 80)
        global_features = self.cnn_global(context)
        global_features = self.pre_g_fc2(global_features.view(-1, 256))
        global_features = self.pre_g_batch_norm2(global_features)
        global_features = F.relu(global_features)

        # context embedding pairs
        context_embeddings_pairs = self.context_combination(context_embeddings)
        context_features = self.deeper_features(context_embeddings_pairs, global_features, 36, keepdim=True)
    
        # context choice embedding pairs
        context_choice_embeddings_pairs = self.context_choice_combination(panel_embeddings)
        context_choice_features = [self.deeper_features(context_choice_embeddings_pairs[:, i, :, :], global_features, 48, keepdim=True) \
                                   for i in range(4)]
        context_choice_features = torch.cat(context_choice_features, dim=1)

        final_features = context_features + context_choice_features
        pred = self.f(final_features.view(-1, 512)).view(-1, 4, 3)
        return pred