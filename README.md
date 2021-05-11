# new_perceiver_pytorch
This model is a modified version of [Perceiver](https://arxiv.org/abs/2103.03206). The cross-attention model and key/value input locations are different from the original paper. 

The transformer unit is a slight modification of the template given by Peter Bloem in his [transformer tutorial](http://peterbloem.nl/blog/transformers). One may change the internal structure on one's own will. It would be a great honour if (s)he shares h(er/is) modification of my code. This model is rather different from the original perceiver model as its cross-attention sections only takes queries and latent tensors where the key and the value tensors are input directly to the transformer blocks with its queries as the output of the cross-attention module.
The model diagram is as follows:
![model](https://github.com/kyunghoon-han/perceiver_pytorch/blob/main/model_diagram.png?raw=true)

## cross_attention.py

This module takes the latent tensor and a query tensor. The design is a modification of a model introduced in [a paper by Ruibing Hou et. al.](https://arxiv.org/pdf/1910.07677v1.pdf).

## transformer.py
The following modules of this file may be of use:
1. device_assigner(tensor_input)  : assigns a device to a tensor or a model
2. attention_block                          :  inputs: q,k,v
3. transformer_block                     :  inputs: q,k,v

## perceiver.py

The main perceiver module. 

## Things to remember
1. Note that it is highly recommended to use the LAMB optimizer for the training.
2. To save some memory issues, the model is designed to take the latent tensor input from the outside of the model.

## Things to be updated
1. incorporate key and value information to the cross attention module.
