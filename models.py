

import torch
from torch import nn
from modules import *
from collections import OrderedDict


class ConvNet(nn.Module):

    def __init__(self, conv_params, linear_params):
        super().__init__()
        self.network = nn.Sequential(OrderedDict(
            conv_blocks=nn.Sequential(*[
                nn.Sequential(OrderedDict(
                    conv2d=BatchNormFoldedQuantizedConv2d(**conv_param),
                    relu=nn.ReLU()
                )) for conv_param in conv_params
            ]),
            flatten=Flatten(),
            unflatten=Unflatten(),
            linear_blocks=nn.Sequential(*[
                nn.Sequential(OrderedDict(
                    conv2d=BatchNormFoldedQuantizedConv2d(**linear_param),
                    relu=nn.ReLU()
                )) for linear_param in linear_params[:-1]
            ]),
            linear_block=nn.Sequential(OrderedDict(
                conv2d=QuantizedConv2d(**linear_params[-1]),
                flatten=Flatten()
            ))
        ))

    def forward(self, images):
        return self.network(images)
