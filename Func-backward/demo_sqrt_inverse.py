
import torch


class sqrt_and_inverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.sum(torch.sqrt(torch.pow(input, 2) - 1)) + torch.nn.Parameter(torch.Tensor(1))
        # output = input
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input_x, = ctx.saved_tensors
        grad_x = grad_output * (torch.div(input_x, torch.sqrt(torch.pow(input_x, 2) - 1)))
        return grad_x


def sqrt_and_inverse_func(input):
    func = sqrt_and_inverse()
    # output = func(input)
    # 如果用上面这句会报错，应该使用 output = func.apply(input)即可
    # RuntimeError: Legacy autograd function with non-static forward method is deprecated.
    # Please use new-style autograd function with static forward method.
    # (Example: https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function)
    # Use func.apply(input) instead of func(input)
    output = func.apply(input)
    return output


x = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float, requires_grad=True)  # tensor

print('开始前向传播')

z = sqrt_and_inverse_func(x)

print('开始反向传播')
z.backward()

print(x.grad)
