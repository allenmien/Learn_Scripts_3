# -*-coding:utf-8-*-
"""
@Time   : 2019-01-17 14:05
@Author : Mark
@File   : encoder_decoder.py
"""
import torch.nn as nn


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        :param encoder:
        :param decoder:
        :param src_embed:
        :param tgt_embed:
        :param generator:
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        :param src:
        :param tgt:
        :param src_mask:
        :param tgt_mask:
        :return:
        """
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        """
        :param src:
        :param src_mask:
        :return:
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        :param memory:
        :param src_mask:
        :param tgt:
        :param tgt_mask:
        :return:
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
