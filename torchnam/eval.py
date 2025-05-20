import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def get_acc(model, data_loader, n_limit=1e10, n_ensemble=1, device=None):
    model = model.eval()

    if device is None:
        device = next(model.parameters()).device

    correct = 0
    total = 0

    for images, labels in data_loader:
        X = images.to(device)
        Y = labels.to(device)

        pres = 0
        for _ in range(n_ensemble):
            pre = model(X)
            if pre.shape[1] > 1:  # CE
                pre = nn.Softmax(dim=1)(pre)
            else:  # BCE
                pre = nn.Sigmoid()(pre)
            pres += pre
        pre = pres/n_ensemble

        if pre.shape[1] > 1:  # CE
            _, pre = torch.max(pre.data, 1)
        else:  # BCE
            pre = (pre > 0.5).squeeze()

        total += pre.size(0)
        correct += (pre == Y).sum()

        if total > n_limit:
            break

    return (100 * float(correct) / total)


@torch.no_grad()
def get_mse(model, data_loader, n_limit=1e10, n_ensemble=1, device=None):
    model = model.eval()

    if device is None:
        device = next(model.parameters()).device

    losses = []
    total = 0

    for images, labels in data_loader:
        X = images.to(device)
        Y = labels.to(device)

        pre = 0
        for _ in range(n_ensemble):
            pre += model(X).squeeze()
        pre = pre / n_ensemble

        loss = nn.MSELoss(reduction='sum')(pre.squeeze(), Y.squeeze()).sum()
        losses.append(loss)

        total += pre.size(0)
        if total > n_limit:
            break

    return torch.tensor(losses).sum()/total


@torch.no_grad()
def get_auc(model, data_loader, n_limit=1e10, n_ensemble=1, device=None):
    model = model.eval()

    if device is None:
        device = next(model.parameters()).device

    pred_total = torch.Tensor([])
    true_total = torch.Tensor([])
    total = 0

    for images, labels in data_loader:

        X = images.to(device)
        Y = labels.to(device)
        true_total = torch.cat((true_total.to(device), Y))

        pres = 0
        for _ in range(n_ensemble):
            pre = model(X)
            if pre.shape[1] > 1:  # CE
                raise ValueError("Not supported for multi-label cases.")
            else:  # BCE
                pre = nn.Sigmoid()(pre)
            pres += pre
        pre = pres/n_ensemble

        pred_total = torch.cat((pred_total.to(device), pre))
        total += pre.size(0)

        if total > n_limit:
            break

    return roc_auc_score(true_total.cpu().detach().numpy(), pred_total.cpu().detach().numpy())