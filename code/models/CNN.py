import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform


class CNN(nn.Module):

    def __init__(self,
                 # dicts,
                 output_dim=100,
                 kernel_size=15,
                 num_filter_maps=128,
                 embedding_dim=300,
                 dropout=0.2,
                 embedding=None
                 ):
        super(CNN, self).__init__()
        torch.manual_seed(42)

        self.output_dim = output_dim
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.embed_drop = nn.Dropout(p=dropout)

        # Make embedding layer:
        if embedding is not None:
            self.embed = nn.Embedding(embedding.size()[0],
                                      embedding.size()[1],
                                      padding_idx=
                                      embedding.size()[0] - 1
                                      )

            self.embed.weight.data = embedding.clone()

        self.conv = nn.Conv1d(self.embedding_dim,
                              num_filter_maps,
                              kernel_size=kernel_size)  # weights
        xavier_uniform(self.conv.weight)

        # Linear output
        self.fc = nn.Linear(num_filter_maps, self.output_dim)
        xavier_uniform(self.fc.weight)

    def forward(self, x, desc_data=None):
        # Embeddings
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        # Conv / Max-pooling
        c = self.conv(x)

        x = F.max_pool1d(F.tanh(c), kernel_size=c.size()[2])
        x = x.squeeze(dim=2)

        # Linear output
        x = self.fc(x)
        return x
