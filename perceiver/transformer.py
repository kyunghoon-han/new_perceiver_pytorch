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


class attention_block(nn.Module):
    def __init__(self, kv_size, seq_length,query_size=100,
                 heads=8,mask=False):
        """
           an attention block 
        """
        super().__init__()
        self.keys = nn.Linear(seq_length, 
                        kv_size * heads, 
                        bias=False).to(device_assigner())
        self.queries = nn.Linear(query_size, 
                        kv_size * heads, 
                        bias=False).to(device_assigner())
        self.values = nn.Linear(seq_length, 
                        kv_size * heads, 
                        bias=False).to(device_assigner())
        
        self.kv = kv_size
        self.heads = heads
        self.mask = mask
        self.q = query_size
        
        self.final_layer = nn.Linear(kv_size * heads, query_size)

    def forward(self, q,k,v):
        v = v.to(torch.float)
        k = self.keys(k)
        v = self.values(v)
        q = self.queries(q)
        # rescale the queries and keys
        q = q / (self.q**(1/4))
        k = k / (self.kv**(1/4))

        # dot product b/w the queries and the keys
        qk = torch.matmul(q,k.transpose(0,1))
        qk = qk.view(qk.size()[0],-1,
                     qk.size()[-1]) # channel_number = 1
        # apply the mask
        if self.mask:
            qk = masker(qk, mask_diag=False)
        
        # apply the softmax
        qk = F.softmax(qk, dim=-1)

        # attention
        att_output = torch.matmul(qk, v)
        return self.final_layer(att_output)

class transformer_block(nn.Module):
    """
        transformer block
    """
    def __init__(self, query_size, kv_dim, 
                 heads,sequence_length, 
                 num_tokens,
                 mask, no_embedding=False):
        super().__init__()
        self.sl = sequence_length
        self.att = attention_block(kv_size=kv_dim,
                                   seq_length=sequence_length,
                                   query_size=query_size,
                                   heads=heads,mask=mask)
        self.no_emb = no_embedding
        self.emb_token = nn.Embedding(embedding_dim=kv_dim,
                                      num_embeddings=num_tokens)
        self.emb_pos = nn.Embedding(embedding_dim=kv_dim,
                                    num_embeddings=num_tokens)
        self.dense_layer = nn.Linear(query_size,query_size)
        self.k = kv_dim
        self.q = query_size
        self.s = sequence_length
    def forward(self, q,k,v):

        k = self.emb_token(k).type(torch.FloatTensor)
        v = self.emb_token(v).type(torch.FloatTensor)
        q = self.emb_token(q).type(torch.FloatTensor)

        positions = self.emb_pos(
                        torch.arange(
                            k.size(-1),
                            device=device_assigner()))

        positions = positions.transpose(
                0,1).expand(k.size()).type(torch.FloatTensor)

        positions = positions.to(device_assigner())

        k = k + positions

        output = self.att(q,k,v)
        output = self.dense_layer(output)
        output = output.view(output.size()[0],1,-1,output.size()[2])
        output = F.log_softmax(output,dim=2)
        return output
