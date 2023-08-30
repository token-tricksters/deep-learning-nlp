import sys

import numpy as np
import torch

from optimizer import SophiaH

# from pytorch_optimizer import SophiaH

seed = 0


def test_optimizer(opt_class) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = opt_class(
        model.parameters(),
        lr=1e-4,
        update_period=11,
        betas=(0.9, 0.995),
        eps=1e-8,
        weight_decay=0.0,
        rho=0.01,
    )
    loss = 1
    for i in range(100):
        opt.zero_grad(set_to_none=True)
        x = torch.FloatTensor(rng.uniform(size=[model.in_features]))
        y_hat = model(x)
        y = torch.Tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward(create_graph=True)
        opt.step()
    print("Final loss:", loss.item(), file=sys.stderr)
    return model.weight.detach()


# ref = test_optimizer(SophiaH)
# np.save("sophia_test.npy", ref.numpy())
# exit(0)

ref = torch.tensor(np.load("sophia_test.npy"))
actual = test_optimizer(SophiaH)
# print weights
print("Reference weights:")
print(ref)
print("Actual weights:")
print(actual)
assert torch.allclose(ref, actual, atol=1e-6, rtol=1e-4)
print("Optimizer test passed!")
