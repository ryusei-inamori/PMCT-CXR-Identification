import math
import torch
from torch import nn
import torch.nn.functional as F


class AdaCos(nn.Module):
    """Checkpoint-compatible AdaCos classification head.

    The trainable parameter name is intentionally `W` to remain compatible
    with the recovered checkpoint (`classifier.metric.W`).
    """

    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.s = math.sqrt(2.0) * math.log(max(2, num_classes - 1))
        self.W = nn.Parameter(torch.empty(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        W = F.normalize(self.W, dim=1)
        logits = F.linear(x, W)
        if labels is None:
            return logits

        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        with torch.no_grad():
            b_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            b_avg = torch.sum(b_avg) / x.size(0)
            theta_med = torch.median(theta[one_hot == 1])
            cap = torch.full_like(theta_med, math.pi / 4)
            self.s = float((torch.log(b_avg) / torch.cos(torch.minimum(cap, theta_med))).item())
        return self.s * logits
