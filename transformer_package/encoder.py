# -*-coding:utf-8-*-
"""
@Time   : 2019-01-17 14:14
@Author : Mark
@File   : encoder.py
"""
import torch.nn as nn

from clone import clones
from layer_norm import LayerNorm


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        """
        :param layer:
        :param N:
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        :param x:
        :param mask:
        :return:
        """
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
