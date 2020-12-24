import torch
import torch.nn as nn
import torch.optim as optim

class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output = output + bias.unsqueeze(0).expand_as(output)
            # 如果是输出是一维的张量的话，且如果加了偏置bias，那么在 backward时候 运行自己定义的那部分 会出错。
            # output = torch.addmm(bias.unsqueeze(0).expand_as(output), input, weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            # grad_bias = grad_output.sum(0).squeeze(0)
            # 如果对于 只是一个 1*1的 tensor的话，squeeze(0)会改变其shape大小
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias)


num_inputs = 2
num_hidden = 5
num_output = 1
num_examples = 4
x = torch.randn(num_examples,num_inputs,dtype=torch.float)
x = torch.tensor(x, requires_grad=True)

# net1 = Linear(num_inputs, num_hidden)
# net2 = Linear(num_hidden,num_output)
# y1 = net1(x)
# y_hat = net2(y1)
# y_hat.sum().backward()

net = nn.Sequential(
    Linear(num_inputs, num_hidden),
    Linear(num_hidden,num_output)
)

y_hat = net(x)

print(list(net.parameters()))
for param in net.parameters():
    print(param)

optimizer = optim.SGD(net.parameters(), lr=1)
optimizer.zero_grad()
y_hat.sum().backward()

print([param.grad for param in net.parameters()])

optimizer.step()

print(list(net.parameters()))

print('testing')

