import torch
from torch import nn

from constants import NLP

F = nn.functional


class RNN(nn.Module):
    def __init__(
            self,
            embedding_dim=128,
            rec_layer_type='lstm',
            num_units=128,
            num_layers=2,
            dropout=0,
            output_dim=2,
            embedding=None
    ):
        super().__init__()
        torch.manual_seed(42)

        self.embedding_dim = embedding_dim
        self.rec_layer_type = rec_layer_type.lower()
        self.num_units = num_units
        self.num_layers = num_layers
        self.dropout = dropout

        self.emb = nn.Embedding(NLP.vocab_size, embedding_dim=self.embedding_dim)

        if embedding is not None:
            self.emb = nn.Embedding.from_pretrained(embedding, freeze=False)

        rec_layer = {'lstm': nn.LSTM, 'gru': nn.GRU}[self.rec_layer_type]
        # We have to make sure that the recurrent layer is batch_first,
        # since sklearn assumes the batch dimension to be the first
        self.rec = rec_layer(
            self.embedding_dim, self.num_units, num_layers=num_layers, batch_first=True)

        self.output = nn.Linear(self.num_units, output_dim)

    def forward(self, X):
        embeddings = self.emb(X)
        # from the recurrent layer, only take the activities from the last sequence step
        if self.rec_layer_type == 'gru':
            _, rec_out = self.rec(embeddings)
        else:
            _, (rec_out, _) = self.rec(embeddings)
        rec_out = rec_out[-1]  # take output of last RNN layer
        drop = F.dropout(rec_out, p=self.dropout)
        # Remember that the final non-linearity should be softmax, so that our predict_proba
        # method outputs actual probabilities!
        out = self.output(drop)
        return out
