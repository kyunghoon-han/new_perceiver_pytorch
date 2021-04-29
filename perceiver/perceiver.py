import torch
from torch import nn
import torch.nn.functional as F
from cross_attention import cross_attention as ca
from transformer import transformer_block as tb
from transformer import device_assigner
import math, random

class perceiver(nn.Module):
    def __init__(self,
                 embedding_size,
                 num_tokens,
                 sequence_length,
                 batch_size,
                 #output_size,
                 dropout=0.1,
                 intermediate_channels=10,
                 bottleneck=16,
                 depth=10,
                 correlation_depth=2,
                 num_heads=20,
                 no_embedding=True):
        super().__init__()
        # cross-attention block
        self.ca = ca(size_features=bottleneck,
                     channels=intermediate_channels,
                     correlation_depth=correlation_depth).to(
                             device_assigner())

        # transformer block
        self.tb = tb(query_size=bottleneck,
                     kv_dim = embedding_size,
                     heads=num_heads,
                     sequence_length=sequence_length,
                     num_tokens=num_tokens,
                     mask=True).to(
                             device_assigner())
        self.depth = depth
        self.rels = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.seqlen = sequence_length
        half_seqlen = int(sequence_length/2)
        self.linear_1 = nn.Linear(bottleneck, half_seqlen)
        self.linear_2 = nn.Linear(bottleneck, half_seqlen)
        self.final = nn.Linear(half_seqlen*half_seqlen,sequence_length*num_tokens)
        self.bs=bottleneck
        self.embedding = nn.Embedding(num_tokens,embedding_size)
    def forward(self, x,latent_tensor): # for now, keep the key and value the same
        #q=(batch,last_output,bs,bs)
        query_tensor = latent_tensor.clone()

        for i in range(self.depth):
            latent_tensor, query_tensor = self.ca(
                    latent_tensor, query_tensor)
            query_tensor = self.tb(
                    query_tensor, x, x)
        output = self.rels(self.dropout(self.linear_1(query_tensor)))
        output = output.view(-1, output.size()[-1],output.size()[-2])
        output = self.rels(self.dropout(self.linear_2(output)))
        output = output.view(output.size()[0],-1)
        output = self.rels(self.dropout(self.final(output)))
        return output
