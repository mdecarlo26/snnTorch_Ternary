import torch
from torch import nn

class ATan(torch.autograd.Function):
    """
    Ternary surrogate gradient.

    Forward:
        sign-like ternary spike:
            +1 if U > 0
             0 if U = 0
            -1 if U < 0

    Backward:
        dS/dU = alpha/2 * 1 / (1 + (pi/2 * alpha * U)^2)
    """

    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.save_for_backward(input_)
        ctx.alpha = alpha
        # ternary sign output in {-1, 0, +1}
        out = (input_ > 0).float() + (input_ < 0).float() * (-1.0)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            ctx.alpha
            / 2
            / (1 + (torch.pi / 2 * ctx.alpha * input_).pow(2))
        )
        return grad * grad_input, None


class ATanSurrogate(nn.Module):
    """
    nn.Module wrapper to match snnTorch's surrogate API.

    Usage:
        spike_grad = ATanSurrogate(alpha=2.0)
        spk = spike_grad(u - theta)
    """

    def __init__(self, alpha: float = 2.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, input_):
        return ATan.apply(input_, self.alpha)


def atan_ternary(alpha: float = 2.0) -> nn.Module:
    """
    Factory function, same pattern as snntorch.surrogate.fast_sigmoid() etc.
    """
    return ATanSurrogate(alpha=alpha)
