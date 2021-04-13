import torch
from torch import nn
import torch.nn.functional as F
import math, random
# import the transformer and attention blocks
from transformer import Transformer, device_assigner
from transformer import cross_attention_block as CA_block

def list_pblocks(depth):
    list_out = []
    for a in range(depth):
        list_out.append(perceiver_block(...))
    return list_out

class Perceiver(nn.Module):
    def __init__(self,
                 embed,
                 num_tokens,
                 sequence_len,
                 bottleneck_size=50,
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
        self.bs = bottleneck_size
        dout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(embedding_dim=embed,
                                      num_embeddings=num_tokens)

        self.transformer = Transformer(
                                emb_size=embed,
                                heads=heads,
                                bottleneck=bottleneck_size,
                                depth=transformer_depth,
                                seq_length=sequence_len,
                                num_tokens=num_tokens,
                                no_embedding=True)
        self.cross_attention = CA_block(embed,heads=heads,
                             bottleneck=bottleneck_size,
                             mask=mask)
        self.last_fc = nn.Linear(embed, self.nt)
        self.to_one = nn.Linear(self.nt,1)
        self.first_run = True
        self.hid = None
        for i in range(self.depth): 
            connector_block.append(fc_unit.to(device_assigner()))
            connector_block.append(relu.to(device_assigner()))
            connector_block.append(dout.to(device_assigner()))
            a_connector = nn.Sequential(*connector_block)
            self.connector_blocks.append(a_connector)
            connector_block = [] 
    def forward(self,x):
        if self.first_run and self.hid is None:
            self.first_run = False
            self.hid = torch.randn(x.size()[0],
                                   self.bs,
                                   self.bs,).to(device_assigner())
        x = self.embedding(x)
        for i in range(self.depth):
            if self.hid is not None:
                x, self.hid = self.cross_attention(x,self.hid)
                x, self.hid = self.transformer(x,self.hid)
            x = self.connector_blocks[i](x)
            x = F.log_softmax(x, dim=2)
        x = self.last_fc(x)
        x = F.log_softmax(x,dim=2)
        return x
