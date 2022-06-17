import torch
import torch.nn as nn

# normalization, pointwise gaussian
class UnitGaussianNormalizer(nn.Module):

    def __init__(self, x, eps=0.00001):
        super().__init__()
        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.register_buffer('mean', torch.mean(x, 0))
        self.register_buffer('std', torch.std(x, 0))
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean