import os

import torch
from torch import nn
from torch.nn import Linear, functional as F, Dropout
from models.transformer import TransformerModel

class AgeTransformer(nn.Module):
    def __init__(self, backbone, transformer, backbone_feature_size, transformer_feature_size):#, num_queries, aux_loss=False):
        super().__init__()

        self.backbone = backbone
        self.transformer = transformer
        self.linear = nn.Linear(backbone_feature_size, transformer_feature_size)

        self.batch_norm = nn.BatchNorm1d(transformer_feature_size)

    def FreezeBaseCnn(self, OnOff):
        for param in self.backbone.parameters():
            param.requires_grad = not OnOff

    def forward(self, input):
        unpacked_input = input.view(input.shape[0] * input.shape[1], input.shape[2], input.shape[3], input.shape[4]) #batch*augment, channel, height, width
        embs = self.backbone(unpacked_input) #Feature Extraction 

        embs = self.linear(embs) 
        embs = F.leaky_relu(embs)

        # embs = self.batch_norm(embs)

        embs = F.normalize(embs, dim=1)

        packed_embs = embs.view(input.shape[0], input.shape[1], -1)
        # packed_embs = embs.reshape((int(embs.shape[0] / input.shape[1]), int(input.shape[1]), embs.shape[1]))

        output = self.transformer(packed_embs)

        return output