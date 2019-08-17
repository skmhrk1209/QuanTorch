import torch
from torch import nn
from modules import *
from distributed import *


class QuantizationEnabler(object):

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        for module in self.model.modules():
            if isinstance(module, QuantizedConv2d):
                module.enable_quantization()

    def __exit__(self, exc_type, exc_value, traceback):
        for module in self.model.modules():
            if isinstance(module, QuantizedConv2d):
                module.disable_quantization()


class BatchStatsUser(object):

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        for module in self.model.modules():
            if isinstance(module, BatchNormFoldedQuantizedConv2d):
                module.use_batch_stats()

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class AverageStatsUser(object):

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        for module in self.model.modules():
            if isinstance(module, BatchNormFoldedQuantizedConv2d):
                module.use_average_stats()

    def __exit__(self, exc_type, exc_value, traceback):
        pass
