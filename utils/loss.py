from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_distance_matrix(x, y, eps=1e-6):
    M, N = x.size(0), y.size(0)
    x2 = torch.sum(x * x, dim=1, keepdim=True).repeat(1, N)
    y2 = torch.sum(y * y, dim=1, keepdim=True).repeat(1, M)
    dist2 = x2 + torch.t(y2) - 2.0 * torch.matmul(x, torch.t(y))
    dist2 = torch.clamp(dist2, min=eps)
    return torch.sqrt(dist2)


def batch_hard_mining(dist_mat, labels):
    assert len(dist_mat.size()) == 2 and len(labels.size()) == 1
    assert dist_mat.size(0) == dist_mat.size(1) == labels.size(0)

    N = dist_mat.size(0)
    labels_NN = labels.view(N, 1).expand(N, N)
    is_pos = labels_NN.eq(labels_NN.t())
    is_neg = labels_NN.ne(labels_NN.t())
    dist_ap, _ = torch.max(torch.reshape(dist_mat[is_pos], (N, -1)), 1, keepdim=False)
    dist_an, _ = torch.min(torch.reshape(dist_mat[is_neg], (N, -1)), 1, keepdim=False)
    return dist_ap, dist_an


def batch_hard_negative_mining(dist_mat):
    M, N = dist_mat.size(0), dist_mat.size(1)
    assert M == N
    labels = torch.arange(N, device=dist_mat.device).view(N, 1).expand(N, N)
    is_neg = labels.ne(labels.t())
    dist_an, _ = torch.min(torch.reshape(dist_mat[is_neg], (N, -1)), 1, keepdim=False)
    return dist_an


def batch_hard_positive_mining(dist_mat, labels):
    assert len(dist_mat.size()) == 2 and len(labels.size()) == 1
    assert dist_mat.size(0) == dist_mat.size(1) == labels.size(0)

    N = dist_mat.size(0)
    labels_NN = labels.view(N, 1).expand(N, N)
    is_pos = labels_NN.eq(labels_NN.t())
    dist_ap, _ = torch.max(torch.reshape(dist_mat[is_pos], (N, -1)), 1, keepdim=False)
    return dist_ap


class BatchHardLoss(nn.Module):

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        if margin is not None:
            self.loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.loss = nn.SoftMarginLoss()

    def forward(self, features, labels):
        dist_mat = pairwise_distance_matrix(features, features)
        dist_ap, dist_an = batch_hard_mining(dist_mat, labels)
        y = torch.ones_like(dist_an)
        if self.margin is not None:
            loss = self.loss(dist_an, dist_ap, y)
        else:
            loss = self.loss(dist_an - dist_ap, y)
        return loss


class BatchHardNegativeLoss(nn.Module):

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        if margin is not None:
            self.loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.loss = nn.SoftMarginLoss()

    def forward(self, anchor_features, positive_features):
        dist_mat = pairwise_distance_matrix(anchor_features, positive_features)
        dist_ap = torch.diagonal(dist_mat)
        dist_an = batch_hard_negative_mining(dist_mat)
        y = torch.ones_like(dist_ap)
        if self.margin is not None:
            loss = self.loss(dist_an, dist_ap, y)
        else:
            loss = self.loss(dist_an - dist_ap, y)
        return loss
