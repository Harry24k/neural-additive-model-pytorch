import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn


class GroupLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, n_features, bias=False, bias_first=False):
        super(GroupLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_features = n_features
        self.group_conv = torch.nn.Conv1d(in_features*n_features, out_features*n_features,
                                          kernel_size=1, bias=False, groups=n_features)
        self.bias_first = bias_first

        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features*n_features))
            torch.nn.init.trunc_normal_(self.bias, std=0.1)
        else:
            self.bias = 0

        if self.bias_first:
            self.bias_first = Parameter(
                torch.Tensor(1, in_features*n_features))
            torch.nn.init.trunc_normal_(self.bias_first, std=0.1)
        else:
            self.bias_first = 0

    def forward(self, inputs):
        batch_size = len(inputs)
        inputs = inputs.view(batch_size, self.in_features *
                             self.n_features) + self.bias_first
        output = self.group_conv(inputs.view(
            batch_size, self.in_features*self.n_features, 1))
        output = output.view(batch_size, self.out_features*self.n_features)
        output += self.bias
        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'


class ResGroupLinear(nn.Module):
    def __init__(self, in_dim, out_dim, n_features, bias):
        super().__init__()
        assert in_dim == out_dim
        h_dim = out_dim
        self.sub = nn.Sequential(
            GroupLinear(in_dim, h_dim, n_features, bias=bias),
            nn.BatchNorm1d(h_dim*n_features),
            nn.ReLU(),
            GroupLinear(in_dim, h_dim, n_features, bias=bias),
            nn.BatchNorm1d(h_dim*n_features),
        )

    def forward(self, input):
        output = self.sub(input)
        return input + output
