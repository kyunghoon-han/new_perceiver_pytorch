# perceiver_pytorch

The transformer unit is a slight modification of the template given by Peter Bloem in his [transformer tutorial](http://peterbloem.nl/blog/transformers). One may change the internal structure on one's own will. It would be a great honour if (s)he shares h(er/is) modification of my code. This model is rather different from the original perceiver model as its cross-attention sections only takes queries and latent tensors where the key and the value tensors are input directly to the transformer blocks with its queries as the output of the cross-attention module.


## cross_attention.py

This module takes the latent tensor and a query tensor. The design is a modification of a model introduced in [a paper by Ruibing Hou et. al.](https://arxiv.org/pdf/1910.07677v1.pdf).

## transformer.py
The following modules of this file may be of use:
1. device_assigner(tensor_input)  : assigns a device to a tensor or a model
2. attention_block                          :  inputs: q,k,v
3. transformer_block                     :  inputs: q,k,v

## perceiver.py

The main perceiver module.

## Things to be updated
1. The model crashes when nn.DataParallel is employed
2. Although I tried to make the model as general as possible, it is still immature.
3. The model is only tested on simple text generation problems, it is not applied to any complex dataset yet. 
4. Basically is full of errors...
