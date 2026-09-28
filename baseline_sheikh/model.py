"""Mạng phân loại hai nhánh theo Sheikh et al. (2023), mục 3.3.

FluentBranch: trôi chảy hay có lỗi (2 lớp).
DisfluentBranch: loại lỗi R/P/B/I (4 lớp); mẫu trôi chảy không tính loss ở nhánh này.
Mỗi nhánh có 3 lớp fully connected; sau mỗi lớp ẩn là ReLU + BatchNorm1d, dropout 0.2 ở hai lớp đầu.
"""
import torch
from torch import nn

import config


def _branch(in_dim, out_dim):
    layers, d = [], in_dim
    for h in config.HIDDEN:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(config.DROPOUT)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class TwoBranchNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fluent = _branch(in_dim, 2)
        self.disfluent = _branch(in_dim, len(config.DISFLUENT))

    def forward(self, x):
        return self.fluent(x), self.disfluent(x)


FLUENT_IDX = config.CLASSES.index("F")


def loss_fn(fluent_logits, disfl_logits, y):
    is_fluent = (y == FLUENT_IDX).long()
    loss_f = nn.functional.cross_entropy(fluent_logits, is_fluent)
    mask = y != FLUENT_IDX
    if mask.any():
        loss_d = nn.functional.cross_entropy(disfl_logits[mask], y[mask])
    else:
        loss_d = disfl_logits.sum() * 0.0
    return loss_f + loss_d


@torch.no_grad()
def predict(model, x):
    model.eval()
    fl, dl = model(x)
    pred = dl.argmax(dim=1)
    pred[fl.argmax(dim=1) == 1] = FLUENT_IDX
    return pred
