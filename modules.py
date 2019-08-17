import torch
from torch import nn
from torch import autograd
from torch import distributed
from distributed import *
from quantizers import *
from range_trackers import *
from enum import *


class Identity(nn.Module):
    def forward(self, inputs):
        return inputs


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, inputs):
        return inputs.reshape(*self.shape)


class Flatten(nn.Module):
    def forward(self, inputs):
        assert inputs.dim() == 4
        return inputs.reshape(inputs.size(0), -1)


class Unflatten(nn.Module):
    def forward(self, inputs):
        assert inputs.dim() == 2
        return inputs.reshape(inputs.size(0), -1, 1, 1)


class QuantizedConv2d(nn.Conv2d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        activation_quantizer=None,
        weight_quantizer=None
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.activation_quantizer = activation_quantizer or AsymmetricQuantizer(
            bits_precision=8,
            range_tracker=AveragedRangeTracker((1, 1, 1, 1))
        )
        self.weight_quantizer = weight_quantizer or AsymmetricQuantizer(
            bits_precision=8,
            range_tracker=GlobalRangeTracker((1, out_channels, 1, 1))
        )

        self.quantization = False

    def enable_quantization(self):
        self.quantization = True

    def disable_quantization(self):
        self.quantization = False

    def forward(self, inputs):

        weight = self.weight
        if self.quantization:
            inputs = self.activation_quantizer(inputs)
            weight = self.weight_quantizer(self.weight)

        outputs = nn.functional.conv2d(
            input=inputs,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

        return outputs


class BatchNormFoldedQuantizedConv2d(QuantizedConv2d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        eps=1e-5,
        momentum=0.1,
        activation_quantizer=None,
        weight_quantizer=None
    ):
        assert bias is False

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            activation_quantizer=activation_quantizer,
            weight_quantizer=weight_quantizer
        )

        self.eps = eps
        self.momentum = momentum

        self.register_parameter('beta', nn.Parameter(torch.zeros(out_channels)))
        self.register_parameter('gamma', nn.Parameter(torch.ones(out_channels)))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))

        self.batch_stats = True

    def use_batch_stats(self):
        self.batch_stats = True

    def use_running_stats(self):
        self.batch_stats = False

    def forward(self, inputs):

        def reshape_to_activation(inputs):
            return inputs.reshape(1, -1, 1, 1)

        def reshape_to_weight(inputs):
            return inputs.reshape(-1, 1, 1, 1)

        def reshape_to_bias(inputs):
            return inputs.reshape(-1)

        if self.training:

            outputs = nn.functional.conv2d(
                input=inputs,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
            dims = [dim for dim in range(4) if dim != 1]
            batch_mean = torch.mean(outputs, dim=dims)
            batch_var = torch.var(outputs, dim=dims)
            batch_std = torch.sqrt(batch_var + self.eps)

            self.running_mean = self.running_mean * (1 - self.momentum) + batch_mean * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + batch_var * self.momentum

        running_mean = self.running_mean
        running_var = self.running_var
        running_std = torch.sqrt(running_var + self.eps)

        weight = self.weight * reshape_to_weight(self.gamma / running_std)
        bias = reshape_to_bias(self.beta - self.gamma * running_mean / running_std)

        if self.quantization:
            inputs = self.activation_quantizer(inputs)
            weight = self.weight_quantizer(weight)

        outputs = nn.functional.conv2d(
            input=inputs,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

        if self.training and self.batch_stats:
            outputs *= reshape_to_activation(running_std / batch_std)
            outputs += reshape_to_activation(self.gamma * (running_mean / running_std - batch_mean / batch_std))

        return outputs
