from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                # State should be stored in this dictionary
                state = self.state[p]

                device = grad.device

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta_1, beta_2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                correct_bias = group["correct_bias"]

                # Init state variables

                if "t" not in state:
                    state["t"] = torch.tensor([0]).to(device)

                if "m" not in state:
                    state["m"] = torch.zeros(size=grad.size(), dtype=grad.dtype).to(
                        device
                    )

                if "v" not in state:
                    state["v"] = torch.zeros(size=grad.size(), dtype=grad.dtype).to(
                        device
                    )

                state["t"] += 1

                # Calculation of new weights

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).

                # 1- Update first and second moments of the gradients

                state["m"] = beta_1 * state["m"] + (1 - beta_1) * grad
                state["v"] = beta_2 * state["v"] + (1 - beta_2) * torch.square(grad)

                # 2- Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                if correct_bias:
                    alpha = (
                        alpha
                        * torch.sqrt(1 - beta_2 ** state["t"])
                        / (1 - beta_1 ** state["t"])
                    )

                # 3- Update parameters (p.data).

                p.data = p.data - alpha * state["m"] / (torch.sqrt(state["v"]) + eps)

                # 4- After that main gradient-based update, update again using weight decay
                #    (incorporating the learning rate again).

                p.data = p.data - group["lr"] * p.data * weight_decay

        return loss


class SophiaG(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.1,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= rho:
            raise ValueError("Invalid rho value: {} - should be >= 0.0".format(rho))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {} - should be >= 0.0".format(weight_decay)
            )
        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
        )
        super(SophiaG, self).__init__(params, defaults)

    def step(self, closure: Callable = None, bs: int = 5120):
        pass
