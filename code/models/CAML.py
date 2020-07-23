from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform


class CAML(nn.Module):

    def __init__(
            self,
            output_dim=200,
            kernel_size=15,
            num_filter_maps=256,
            embedding_dim=300,
            dropout=0.2,
            embedding=None):
        super(CAML, self).__init__()
        torch.manual_seed(42)

        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.embed_drop = nn.Dropout(p=dropout)

        if embedding is not None:
            self.embed = nn.Embedding(embedding.size()[0],
                                      embedding.size()[1],
                                      padding_idx=
                                      embedding.size()[0] - 1
                                      )
            self.embed.weight.data = embedding.clone()

        # Initialize convolution layer as in 2.1
        self.conv = nn.Conv1d(self.embedding_dim,
                              num_filter_maps,
                              kernel_size=kernel_size,
                              padding=int(floor(kernel_size / 2)))

        xavier_uniform(self.conv.weight)

        # Context vectors for computing attention as in 2.2
        self.U = nn.Linear(num_filter_maps, output_dim)
        xavier_uniform(self.U.weight)

        # Final layer: create a matrix to use for the L binary
        # classifiers as in 2.3
        self.final = nn.Linear(num_filter_maps, output_dim)
        xavier_uniform(self.final.weight)

    def forward(self, x, desc_data=None, get_attention=True):
        # Get embeddings and apply dropout
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        # Apply convolution and non-linearity (tanh)
        x = F.tanh(self.conv(x).transpose(1, 2))

        # Apply attention
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)

        # Document representations are weighted sums using the attention.
        # Can compute all at once as a matmul
        m = alpha.matmul(x)

        # Final layer
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        return y
