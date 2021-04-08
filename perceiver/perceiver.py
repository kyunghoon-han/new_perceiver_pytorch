import torch
from torch import nn
import torch.nn.functional as F
import math, random
# import the transformer and attention blocks
from transformer import Transformer, device_assigner
from transformer import self_attention_block as SA_block

class Perceiver(nn.Module):
    def __init__(self,
                 embed,
                 num_tokens,
                 sequence_len,
                 mask=True,
                 heads=8, depth=3,
                 transformer_depth=2,
                 dropout=0.2):
        super().__init__()
        self.depth = depth
        self.embed = embed
        self.nt = num_tokens
        connector_block = []
        self.connector_blocks = []
        fc_unit = nn.Linear(num_tokens, embed)
        relu = nn.ReLU()
        self.relu = relu
        dout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(embedding_dim=embed,
                                      num_embeddings=num_tokens)

        self.transformer = Transformer(
                                emb_size=embed,
                                heads=heads,
                                depth=transformer_depth,
                                seq_length=sequence_len,
                                num_tokens=num_tokens,
                                no_embedding=True)
        self.self_attention = SA_block(embed,heads=heads,
                             mask=mask)
        self.last_fc = nn.Linear(embed, self.nt)
        for i in range(self.depth): 
            connector_block.append(fc_unit.to(device_assigner()))
            connector_block.append(relu.to(device_assigner()))
            connector_block.append(dout.to(device_assigner()))
            a_connector = nn.Sequential(*connector_block)
            self.connector_blocks.append(a_connector)
            connector_block = [] 
    def forward(self,x):
        x = self.embedding(x)
        for i in range(self.depth):
            x = self.self_attention(x)
            #x = x.view(-1,self.embed*self.nt)
            #x = self.att_to_trans[i](x)
            x = self.transformer(x)
            x = self.connector_blocks[i](x)
        x = self.relu(self.last_fc(x))
        return x

