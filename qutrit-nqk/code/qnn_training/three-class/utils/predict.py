import torch

def predict_Sz_3class(output):
    pred = torch.full_like(output, 1, dtype=torch.long)  # default: class 1
    pred = torch.where(output >  1/3, torch.tensor(0, device=output.device), pred)
    pred = torch.where(output < -1/3, torch.tensor(2, device=output.device), pred)
    return pred

def predict_logits(output):
    return output.argmax(dim=1)

def predict_fidelity(output):
    return output.argmax(dim=1)

