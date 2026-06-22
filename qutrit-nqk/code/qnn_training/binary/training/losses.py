import torch


def cost_Lz_Binary(output, target):
    Y = torch.where(target == 1, torch.tensor(1.0), torch.tensor(-1.0))
    return torch.mean((output - Y)**2)


def cost_fidelity(output, target):
    return torch.mean((1 - output[range(len(target)), target]) ** 2)