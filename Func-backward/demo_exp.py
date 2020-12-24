
import torch

class Exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        resut = torch.exp(i)
        ctx.save_for_backward(resut)
        return resut

    @staticmethod
    def backward(ctx, grad_out):
        result, = ctx.saved_tensors
        return grad_out * result

exp = Exp()

x1 = torch.tensor([1.5, 1.2], requires_grad=True)

x2 = exp.apply(x1)

y3 = exp.apply(x2)

y3.backward(torch.tensor([1., 1.]))
print(x1.grad)
print(x2.grad)
