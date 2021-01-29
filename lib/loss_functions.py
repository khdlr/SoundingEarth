import torch

BCE = torch.nn.functional.binary_cross_entropy_with_logits

def Hinge(prediction, target):
    """Hinge Loss that takes a target \in [0, 1]"""
    with torch.no_grad():
        pm_target = 2 * target - 1
    return torch.mean(F.relu(1. - prediction * target))

