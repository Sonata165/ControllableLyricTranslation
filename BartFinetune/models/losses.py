import os
import sys

import torch.nn as nn
import torch.nn.functional as F

def _main():
    pass


def _procedures():
    pass


class LenRegLoss(nn.Module):
    def __init__(self, consider_index=[1,2], reduction: str = 'mean', label_smoothing = 0.0):
        super().__init__()
        self.consider_index = consider_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, input, target):

        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)


if __name__ == '__main__':
    _main()
