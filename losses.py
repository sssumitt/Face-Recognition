import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    def __init__(self, n_classes, s=64.0, m=0.3, eps=1e-7):
        super().__init__()
        self.s, self.m, self.eps = s, m, eps
        self.ce = nn.CrossEntropyLoss()
        self.n_classes = n_classes

    def forward(self, emb, weight, labels):
        w_norm = F.normalize(weight, dim=1)
        cos_theta = F.linear(emb, w_norm).clamp(-1 + self.eps, 1 - self.eps)
        theta_m = torch.acos(cos_theta) + self.m
        cos_theta_m = torch.cos(theta_m)
        one_hot = F.one_hot(labels, self.n_classes).to(cos_theta.dtype)
        logits = one_hot * cos_theta_m + (1 - one_hot) * cos_theta
        logits *= self.s
        return self.ce(logits, labels)
