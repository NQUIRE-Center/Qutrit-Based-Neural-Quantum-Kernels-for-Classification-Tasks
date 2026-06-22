import torch

 
def cost_Sz_3class(output, target):
    exp_Sz = output
    target = target.to(device=exp_Sz.device)
 
    target_values = torch.empty_like(exp_Sz)
    target_values[target == 0] =  1.0
    target_values[target == 1] =  0.0
    target_values[target == 2] = -1.0
 
    loss = (exp_Sz - target_values) ** 2
    return loss.mean()

def cross_entropy(probs, targets):
    return torch.nn.functional.nll_loss(torch.log(probs), targets)

def cost_fidelity(output, target):
    return torch.mean((1 - output[range(len(target)), target]) ** 2)