# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F


def calculate_acc(output, target):
    output = output.view(-1, 3)
    target = target.view(-1)
    pred = output.data.max(1)[1]
    correct = pred.eq(target.data).cpu().sum().numpy()
    return correct * 100.0 / target.size()[0]


def calculate_correct(output, target):
    output = output.view(-1, 3)
    target = target.view(-1)
    pred = output.data.max(1)[1]
    matrix = pred.eq(target.data).cpu()
    correct = matrix.sum().numpy()
    return correct, matrix
    

def cross_entropy_loss(output, target):
    output = output.view(-1, 3)
    target = target.view(-1)

    return F.cross_entropy(output, target)