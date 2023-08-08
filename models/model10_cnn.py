import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models.resnet import ResNet, BasicBlock
from models.densetcn import DenseTemporalConvNet
from models.swish import Swish

def get_densetcn_options():
    densetcn_options = {
        "relu_type": "swish",
        "width_mult": 1.0,
        "densetcn_block_config": [
            3,
            3,
            3,
            3
        ],
        "densetcn_growth_rate_set": [
            384,
            384,
            384,
            384
        ],
        "densetcn_kernel_size_set": [
            3,
            5,
            7
        ],
        "densetcn_dilation_size_set": [
            1,
            2,
            5
        ],
        "densetcn_reduced_size": 512,
        "densetcn_se": True,
        "densetcn_dropout": 0.2,
        }
    return densetcn_options


# -- auxiliary functions
def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)


def _average_batch(x, lengths):
    return torch.stack( [torch.mean( x[index][:,0:i], 1 ) for index, i in enumerate(lengths)],0 )

class DenseTCN(nn.Module):
    def __init__(self, word_ct_max, word_dim, block_config, growth_rate_set, input_size, reduced_size, kernel_size_set, dilation_size_set, dropout, relu_type, squeeze_excitation=False):
        super(DenseTCN, self).__init__()

        self.word_ct_max = word_ct_max
        self.word_dim = word_dim

        num_features = reduced_size + block_config[-1]*growth_rate_set[-1]
        self.tcn_trunk = DenseTemporalConvNet( block_config, growth_rate_set, input_size, reduced_size,
                                          kernel_size_set, dilation_size_set,
                                          dropout=dropout, relu_type=relu_type,
                                          squeeze_excitation=squeeze_excitation,
                                          )
        self.tcn_output = nn.Linear(num_features, 512)

    def forward(self, x):
        x = self.tcn_trunk(x.transpose(1, 2))
        # print(f"tcn_trunk : {x.shape}")

        x = self.tcn_output(x.transpose(1, 2))
        # print(f"tcn_output : {x.shape}")

        return x

class Lipreading(nn.Module):
    def __init__( self, relu_type='prelu', word_ct_max=71, word_dim=768):
        super(Lipreading, self).__init__()
        self.densetcn_options = get_densetcn_options()
        self.word_ct_max = word_ct_max
        self.word_dim = word_dim

        self.frontend_nout = 64
        self.backend_out = 512
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)

        # -- frontend3D
        if relu_type == 'relu':
            frontend_relu = nn.ReLU(True)
        elif relu_type == 'prelu':
            frontend_relu = nn.PReLU( self.frontend_nout )
        elif relu_type == 'swish':
            frontend_relu = Swish()

        self.frontend3D = nn.Sequential(
                    nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                    nn.BatchNorm3d(self.frontend_nout),
                    frontend_relu,
                    nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))

        self.tcn =  DenseTCN( block_config=self.densetcn_options['densetcn_block_config'],
                                growth_rate_set=self.densetcn_options['densetcn_growth_rate_set'],
                                input_size=self.backend_out,
                                reduced_size=self.densetcn_options['densetcn_reduced_size'],
                                kernel_size_set=self.densetcn_options['densetcn_kernel_size_set'],
                                dilation_size_set=self.densetcn_options['densetcn_dilation_size_set'],
                                dropout=self.densetcn_options['densetcn_dropout'],
                                relu_type=relu_type,
                                squeeze_excitation=self.densetcn_options['densetcn_se'],
                                word_ct_max=self.word_ct_max,
                                word_dim=self.word_dim
                            )

        # -- initialize
        self._initialize_weights_randomly()


    def forward(self, x):
        B, C, T, H, W = x.size()
        x = self.frontend3D(x)
        # print(f"frontend3D : {x.shape}")
        Tnew = x.shape[2]    # outpu should be B x C2 x Tnew x H x W
        x = threeD_to_2D_tensor( x )
        # print(f"threeD_to_2D_tensor : {x.shape}")
        x = self.trunk(x)
        # print(f"trunk : {x.shape}")
        x = x.view(B, Tnew, x.size(1))
        # print(f"view : {x.shape}")

        return self.tcn(x)


    def _initialize_weights_randomly(self):
        use_sqrt = True

        if use_sqrt:
            def f(n):
                return math.sqrt( 2.0/float(n) )
        else:
            def f(n):
                return 2.0/float(n)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod( m.kernel_size ) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))