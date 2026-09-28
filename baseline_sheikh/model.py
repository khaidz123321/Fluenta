"""Mạng hai nhánh của Sheikh et al. (2023), mục 3.3.

Bài nêu ba FC mỗi nhánh, ReLU/BN, dropout 0.2 ở hai FC đầu và tổng hai
cross-entropy. Số neuron ẩn không được công bố; truyền vào từ train_eval.
"""

import torch
from torch import nn

import config

FLUENT_IDX = config.CLASSES.index("F")


def branch(in_dim, out_dim, hidden):
    layers = []
    current = in_dim
    for width in hidden:
        layers.extend((
            nn.Linear(current, width), nn.ReLU(), nn.BatchNorm1d(width),
            nn.Dropout(config.NN_DROPOUT),
        ))
        current = width
    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


class TwoBranchNet(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        if len(hidden) != 2:
            raise ValueError("Mỗi nhánh cần hai lớp ẩn và một lớp ra")
        self.fluent = branch(in_dim, 2, hidden)
        self.disfluent = branch(in_dim, len(config.DISFLUENT), hidden)

    def forward(self, features):
        return self.fluent(features), self.disfluent(features)


def loss_fn(fluent_logits, disfluent_logits, targets):
    binary_targets = (targets == FLUENT_IDX).long()
    loss_f = nn.functional.cross_entropy(fluent_logits, binary_targets)
    mask = targets != FLUENT_IDX
    loss_d = (nn.functional.cross_entropy(disfluent_logits[mask], targets[mask])
              if mask.any() else disfluent_logits.sum() * 0.0)
    return loss_f + loss_d


@torch.no_grad()
def probabilities(model, features):
    """P(F) và P(R/P/B/I) theo xác suất có điều kiện của hai nhánh."""
    model.eval()
    fluent_logits, disfluent_logits = model(features)
    fluent = torch.softmax(fluent_logits, dim=1)
    disfluent = torch.softmax(disfluent_logits, dim=1)
    probs = torch.cat((fluent[:, :1] * disfluent, fluent[:, 1:2]), dim=1)
    return probs


@torch.no_grad()
def predict(model, features):
    """Quy tắc cứng trong bài: nhánh fluent quyết định trước."""
    model.eval()
    fluent_logits, disfluent_logits = model(features)
    predictions = disfluent_logits.argmax(dim=1)
    predictions[fluent_logits.argmax(dim=1) == 1] = FLUENT_IDX
    return predictions
