import torch
from torch import nn
import torch.nn.functional as F
from transformer import device_assigner

# Correlation map
class correlation_map(nn.Module):
    """
        a miniature correlation model
    """
    def __init__(self, 
                 size_features = 100,
                 in_channels = 1,
                 out_channels = 10,
                 kernel_size = 3,
                 dropout=True,
                 depth=2):
        """
            size_features : input and output size
            in_channels   : input channel size
            out_channels  : output channel size
            kernel_size   : kernel size for the conv layers
            dropout       : Boolean to decide whether to 
                            apply the drop out layers
            depth         : depth of the lin/conv layers
        """
        super().__init__()
        linear = nn.Linear(size_features*2,
                           size_features*2).to(device_assigner())
        self.list_lin = nn.ModuleList().to(device_assigner())
        self.batchnorm = nn.BatchNorm2d(out_channels).to(device_assigner())
        for i in range(depth):
            self.list_lin.append(linear)
        self.do = nn.Dropout(0.2)
        self.dropout = dropout
        self.rels = nn.ReLU()
    def forward(self, x, y):
        z = torch.cat((x,y),dim=-1)
        # the inputs must be of size
        # [num_batchs, channel_size, width, height]
        assert len(z.size()) == 4
        for i in range(len(self.list_lin)):
            z = self.list_lin[i](z)
            if self.dropout:
                z = self.do(z)
            z = self.rels(z)
        x,y = z.chunk(2,dim=-1)
        return x,y
        
class internal_dense(nn.Module):
    def __init__(self, feature_size,
                 in_channels=10,
                 out_channels=1,
                 padding=2,
                 kernel_size=3):
        """
            feature_size : size of the tensor input
            in_channels  : input channel size
            out_channels : output channel size
            kernel_size  : size of the kernel
        """
        super().__init__()
        self.lin_p = nn.Linear(feature_size,feature_size)
        self.lin_q = nn.Linear(feature_size,feature_size)
        self.combine = nn.Linear(feature_size*2,feature_size*2)
        self.rels = nn.ReLU()
        self.out = out_channels
    def forward(self,x,y):
        x = self.rels(self.lin_p(x))
        y = self.rels(self.lin_q(y))
        z = torch.cat((x,y),dim=-1)
        z = self.rels(self.combine(z))
        return z.chunk(2,dim=-1)

class cross_attention(nn.Module):
    """
        a small cross attention module
    """
    def __init__(self, size_features = 100, channels=10,
                 kernel_size=3, dropout=True,
                 correlation_depth=2,
                 ):
        """
            corel_map : above-mentioned correlation_map
            int_dense : above-mentioned internal dense map
        """
        super().__init__()
        self.corel_map = correlation_map(size_features=size_features,
                                         out_channels=channels,
                                         kernel_size=kernel_size,
                                         dropout=dropout,
                                         depth=correlation_depth).to(
                                             device_assigner())
        self.int_dense = internal_dense(feature_size=size_features,
                                        in_channels=channels,
                                        kernel_size=kernel_size).to(
                                                device_assigner())
    def forward(self, x, y):
        """
            x : input tensor
            y : query tensor
        """
        x1, y1 = x.clone().detach(), y.clone().detach()
        x, y = self.corel_map(x,y)
        x, y = self.int_dense(x,y)
        x = torch.mul(x,x1)
        y = torch.mul(y,y1)
        return x, y
