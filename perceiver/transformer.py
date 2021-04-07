import torch
from torch import nn
import torch.nn.functional as F
import random, math

def device_assigner(tensor=None):
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

def batch_incorporation(a_tensor):
    shape = a_tensor.size()
    a_tensor = a_tensor.transpose(1,2)
    a_tensor = a_tensor.contiguous()
    a_tensor = a_tensor.view(shape[0]*shape[2],
                    shape[1], shape[3])
    return a_tensor

def masker(matrix, replace_val=0.0,
           value_replaced=0.0, mask_diag=True):
    batchs, height, width = matrix.size()
    if mask_diag:
        offset = 1
    else:
        offset = 0
    indices = torch.triu_indices(
                matrix.size()[1],matrix.size()[2],offset=offset)
    matrix[:, indices[0], indices[1]]= value_replaced
    return matrix


class self_attention_block(nn.Module):
    def __init__(self, embed, heads=8,mask=False):
        """
           an attention block 
        """
        super().__init__()
        self.keys = nn.Linear(embed, embed * heads, bias=False)
        self.queries = nn.Linear(embed, embed * heads, bias=False)
        self.values = nn.Linear(embed, embed * heads, bias=False)
        
        self.embed = embed
        self.heads = heads
        self.mask = mask

        self.final_layer = nn.Linear(embed * heads, embed)

    def forward(self, x):
        width, breadth, height = x.size()
        # height is the embedding dimension
        # this embedding dimension must match the layer size
        assert height == self.embed
        heads = self.heads
        # get the keys, queries and values
        key_vals = self.keys(x).view(
                    width, breadth, heads, height)
        query_vals =self.queries(x).view(
                    width, breadth, heads, height)
        value_vals = self.values(x).view(
                    width, breadth, heads, height)
        # step 1: heads are now incorporated in batches
        #         batch_size = width above * # heads
        key_vals = batch_incorporation(key_vals)
        query_vals = batch_incorporation(query_vals)
        value_vals = batch_incorporation(value_vals)
        # step 2: rescale the queries and the keys
        query_vals = query_vals / (height ** (1/4))
        key_vals = key_vals / (height ** (1/4))
        # step 3: dot product b/w the queries and the keys
        q_and_k = torch.bmm(query_vals, key_vals.transpose(1,2))
        # the size of q_and_k must satisfy the following
        assert q_and_k.size() == (width * heads, breadth, breadth)

        # step 3.5: if mask==True, apply the mask
        if self.mask:
            q_and_k = masker(q_and_k, mask_diag=False)
        # step 4 : apply the softmax function
        q_and_k = F.softmax(q_and_k, dim=2)
        
        # step 5 : apply the self-attention
        att_output = torch.bmm(q_and_k, value_vals)
        att_output = att_output.view(width, heads, breadth, height)
        
        # step 6 : combine the results 
        output = att_output.transpose(1,2)
        output = output.contiguous()
        output = output.view(width, breadth, heads*height)
        output = self.final_layer(output)

        # output the result
        return output

class T_block(nn.Module):
    """
        A transformer block
    """
    def __init__(self, embed, sequence_len,
                 mask=True, head=8, 
                 hidden_size=4, dropout=0.2):
        super().__init__()
        self.attention = self_attention_block(embed, 
                                         heads=head, 
                                         mask=mask)
        self.mask = mask
        # apply the layer normalization
        # y = \gamma*(x - E[x])/\sqrt(Var[x]+\epsilon)+\beta
        self.normal_1 = nn.LayerNorm(embed)
        self.normal_2 = nn.LayerNorm(embed)
        # dropouts
        self.drops = nn.Dropout(dropout)
        # fully_connected layers
        self.fc = nn.Sequential(
                nn.Linear(embed, hidden_size * embed),
                nn.ReLU(),
                nn.Linear(hidden_size * embed, embed)
        )
    def forward(self, x):
        attention = self.attention(x)
        x = self.normal_1(attention + x)
        x = self.drops(x)
        fc_embed = self.fc(x)
        x = self.normal_2(fc_embed + x)
        x = self.drops(x)
        return x

class Transformer(nn.Module):
    """
        a transformer 
        made for generating a set of texts
            - almost an exact copy from the tutorial by
              Peter Bloem
    """
    def __init__(self, emb_size, heads, depth,
                 seq_length, num_tokens,no_embedding=False):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_embed = nn.Embedding(embedding_dim=emb_size,
                                        num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb_size,
                                          num_embeddings=num_tokens)
        self.no_embedding = no_embedding
        transformer_blocks = []
        for i in range(depth):
            transformer_blocks.append(
                    T_block(embed=emb_size, 
                            sequence_len=seq_length,
                            head=heads))
        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.to_outputs = nn.Linear(emb_size, num_tokens)
    def forward(self, x):
        if self.no_embedding:
            tokens=x
        else:
            tokens = self.token_embed(x)
        batches, width, breadth = tokens.size()
        a = torch.arange(breadth, device=device_assigner())
        
        positions = self.pos_embedding(
                            torch.arange(
                                width,
                                device=device_assigner()))
        positions = positions.expand(batches, width, breadth)
        x = tokens + positions
        x = self.transformer_blocks(x)
        x = x.view(batches * width, breadth)
        x = self.to_outputs(x).view(batches, width, self.num_tokens)
        x = F.log_softmax(x, dim=2)
        return x
