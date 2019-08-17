import torch
from torch import nn
from distributed import *


class RangeTracker(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.register_buffer('min_val', None)
        self.register_buffer('max_val', None)

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, inputs):

        keep_dims = [dim for dim, size in enumerate(self.shape) if size != 1]
        reduce_dims = [dim for dim, size in enumerate(self.shape) if size == 1]
        permute_dims = [*keep_dims, *reduce_dims]
        repermute_dims = [permute_dims.index(dim) for dim, size in enumerate(self.shape)]

        inputs = inputs.permute(*permute_dims)
        inputs = inputs.reshape(*inputs.shape[:len(keep_dims)], -1)

        min_val = torch.min(inputs, dim=-1, keepdim=True)[0]
        min_val = min_val.reshape(*inputs.shape[:len(keep_dims)], *[1] * len(reduce_dims))
        min_val = min_val.permute(*repermute_dims)

        max_val = torch.max(inputs, dim=-1, keepdim=True)[0]
        max_val = max_val.reshape(*inputs.shape[:len(keep_dims)], *[1] * len(reduce_dims))
        max_val = max_val.permute(*repermute_dims)

        min_val = torch.min(torch.stack(list(all_gather(min_val))))
        max_val = torch.max(torch.stack(list(all_gather(max_val))))

        self.update_range(min_val, max_val)


class GlobalRangeTracker(RangeTracker):

    def __init__(self, shape):
        super().__init__(shape)

    def update_range(self, min_val, max_val):
        self.min_val = torch.min(self.min_val, min_val) if self.min_val is not None else min_val
        self.max_val = torch.max(self.max_val, max_val) if self.max_val is not None else max_val


class AveragedRangeTracker(RangeTracker):

    def __init__(self, shape, momentum=0.1):
        super().__init__(shape)
        self.momentum = momentum

    def update_range(self, min_val, max_val):
        self.min_val = self.min_val * (1 - self.momentum) + min_val * self.momentum if self.min_val is not None else min_val
        self.max_val = self.max_val * (1 - self.momentum) + max_val * self.momentum if self.max_val is not None else max_val
