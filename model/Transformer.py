import torch
import torch.nn as nn
import torch.nn.unctional as F
import pytorch_lightning as pl

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):


class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):

class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return x

class ViTDecoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):


class ViT(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = ViTEncoder()
        self.decoder = ViTDecoder()

    def forward(self, x, normal):
        enc_out = self.encoder(x)
        out = self.decoder(enc_out, normal)
        return out