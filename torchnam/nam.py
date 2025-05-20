import torch
import torch.nn as nn

from .eval import get_acc, get_auc, get_mse


class NAM(nn.Module):
    def __init__(self, feature_model, n_features, n_classes, n_tasks=1):
        super().__init__()
        self.feature_model = feature_model
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_tasks = n_tasks

        self.linears = []
        for _ in range(self.n_tasks):
            self.linears.append(
                nn.Linear(self.n_features, self.n_classes, bias=True))
        self.linears = nn.ModuleList(self.linears)

    def forward(self, inputs, n_task=0):
        batch_size = len(inputs)
        inputs = inputs.view(batch_size, -1)
        hidden_features = self.feature_model(inputs)
        outputs = self.linears[n_task](hidden_features)
        return outputs

    def forward_with_feature(self, inputs, n_task=0):
        batch_size = len(inputs)
        inputs = inputs.view(batch_size, -1)
        hidden_features = self.feature_model(inputs)

        linear = self.linears[n_task]
        weight = linear.weight
        w = weight.reshape(1, self.n_classes, self.n_features)
        hidden = hidden_features.reshape(batch_size, 1, self.n_features)

        b = linear.bias
        b = b.reshape(self.n_classes)

        hidden = (w*hidden)
        bias = b
        out = hidden.sum(dim=-1) + b.reshape(1, self.n_classes)
#         'hidden':'batch_size x n_clases x n_features',
#         'bias':'n_clases',
#         'out':'batch_size x n_classes',
        return hidden, bias, out

    @torch.no_grad()
    def feature_contribution(self, start, end, step, n_task=0):
        output = {}
        x = torch.arange(start, end+step, step)
        x = torch.stack([x]*self.n_features, dim=1)
        hidden, bias, out = self.forward_with_feature(x, n_task)
        output['shape'] = {
            'hidden': 'batch_size x n_clases x n_features',
            'bias': 'n_clases',
            'out': 'batch_size x n_classes',
        }
        output['hidden'] = hidden
        output['bias'] = bias
        output['out'] = out
        return output

    def get_acc(self, loader, n_ensemble=1):
        return get_acc(self, loader, n_ensemble=n_ensemble)

    def get_auc(self, loader, n_ensemble=1):
        return get_auc(self, loader, n_ensemble=n_ensemble)

    def get_mse(self, loader, n_ensemble=1):
        return get_mse(self, loader, n_ensemble=n_ensemble)
