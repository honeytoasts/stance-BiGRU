# 3rd-party module
import argparse
import torch
from torch import nn
from torch.nn import functional as F

class BaseModel(torch.nn.Module):
    def __init__(self,
                 device: torch.device,
                 config: argparse.Namespace,
                 num_embeddings: int,
                 padding_idx: int,
                 embedding_weight=None):
        super(BaseModel, self).__init__()

        # device
        self.device = device

        # config
        self.config = config

        # embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=num_embeddings,
                                            embedding_dim=config.embedding_dim,
                                            padding_idx=padding_idx)
        if embedding_weight is not None:
            self.embedding_layer.weight = nn.Parameter(embedding_weight)

        # RNN dropout
        self.rnn_dropout = nn.Dropout(p=config.rnn_dropout)

        # parameters of GRU
        parameters = {'input_size': config.embedding_dim,
                      'hidden_size': config.hidden_dim,
                      'num_layers': config.num_rnn_layers,
                      'batch_first': True,
                      'dropout': config.rnn_dropout
                                 if config.num_rnn_layers > 1 else 0,
                      'bidirectional': True}

        # target BiGRU
        self.target_BiGRU = nn.GRU(**parameters)

        # claim BiGRU
        self.claim_BiGRU = nn.GRU(**parameters)

        # linear transformation for target
        self.t_linear = nn.Linear(in_features=2*config.hidden_dim,
                                  out_features=2*config.hidden_dim)

        # linear transformation for claim
        self.c_linear = nn.Linear(in_features=2*config.hidden_dim,
                                  out_features=2*config.hidden_dim)

        # linear transformation for attention score
        self.attn_linear = nn.Linear(in_features=2*config.hidden_dim,
                                     out_features=1)

        # linear dropout
        self.linear_dropout = nn.Dropout(p=config.linear_dropout)

        # linear layer
        self.linear = nn.Linear(in_features=2*config.hidden_dim,
                                out_features=config.output_dim)

    def forward(self, batch_target, batch_claim):
        # embedding layer
        batch_target = self.embedding_layer(batch_target)
        batch_claim = self.embedding_layer(batch_claim)

        # dropout
        batch_claim = self.rnn_dropout(batch_claim)

        # BiGRU
        batch_target, _ = self.target_BiGRU(batch_target)  # (B, S, H)
        batch_target = batch_target[:, -1]  # (B, H)

        batch_claim, _ = self.claim_BiGRU(batch_claim)  # (B, S, H)

        # linear transformation
        e = torch.tanh(self.t_linear(batch_target).unsqueeze(1) +   # (B, 1, H)
                       self.c_linear(batch_claim))  # (B, S, H)

        # get attention score
        weight = self.attn_linear(e).squeeze(2)  # (B, S)
        weight = torch.nn.functional.softmax(weight, dim=1)  # (B, S)

        # get linear combination vector
        r = torch.matmul(weight.unsqueeze(1), batch_claim).squeeze(1)  # (B, H)

        # dropout
        output = self.linear_dropout(r)

        # final linear
        output = self.linear(output)

        return output