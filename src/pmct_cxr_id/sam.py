from __future__ import annotations
import torch
from torch import nn


class SAM(torch.optim.Optimizer):
    """Two-step SAM/ASAM optimizer wrapper used by the public pretraining script."""
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        if rho <= 0:
            raise ValueError("rho must be positive")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e = (p.pow(2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e)
                self.state[p]["e_w"] = e
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise RuntimeError("Use first_step() and second_step() for SAM")

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    scale = torch.abs(p) if group["adaptive"] else 1.0
                    norms.append((scale * p.grad).norm(p=2).to(shared_device))
        return torch.norm(torch.stack(norms), p=2)


def disable_running_stats(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0


def enable_running_stats(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
