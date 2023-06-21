import torch
import numpy as np
from optimizer import SophiaG

seed = 0


def test_optimizer(opt_class) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = opt_class(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    hess_interval = 10
    for i in range(1000):
        opt.zero_grad()
        x = torch.FloatTensor(rng.uniform(size=[model.in_features]))
        y_hat = model(x)
        y = torch.Tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()
        if hasattr(opt, "update_hessian") and i % hess_interval == hess_interval - 1:
            # Update the Hessian EMA
            opt.zero_grad()
            x = torch.FloatTensor(rng.uniform(size=[model.in_features]))
            y_hat = model(x)
            y = torch.Tensor([x[0] + x[1], -x[2]])
            loss = ((y - y_hat).abs()).sum()
            loss.backward()
            opt.update_hessian(bs=model.in_features)
            opt.zero_grad(set_to_none=True)
    return model.weight.detach()


ref = torch.tensor(np.load("sophia_test.npy"))
actual = test_optimizer(SophiaG)
# print weights
print("Reference weights:")
print(ref)
print("Actual weights:")
print(actual)
assert torch.allclose(ref, actual, atol=1e-6, rtol=1e-4)
print("Optimizer test passed!")

if __name__ == "__main__":
    pass
