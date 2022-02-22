# -*- coding: utf-8 -*-
# ---------------------
# code from https://blog.csdn.net/FY_2018/article/details/119973349

import os
import torch
from torch.autograd import Variable
import torch.optim as optim
dir_path = os.path.dirname(__file__)


batchsize = 16
batchnum = 100
# rawdata = torch.randn([batchsize*batchnum,3])
# torch.save(rawdata, 'debug_running_var_mean_rawdata.pth')
#加载数据，保证数据是不变的
rawdata = torch.load('./debug_running_var_mean_rawdata.pth')
print(rawdata.size())
 
y = Variable(torch.FloatTensor([4,5]))
dataset = [Variable(rawdata[curpos:curpos + batchsize]) for curpos in range(0,len(rawdata),batchsize)]
 
 
class MyBatchnorm1d(torch.nn.Module):
    def __init__(self,num_features,momentum=0.9):
        '''
        自定义的batchnorm
        :param num_features:
        :param momentum: 动量系数,大于等于0小于1,表示保留原来变量值的比例，与标准库torch.nn.Batchnorm1d正好相反
                         当取None时，采用简单的取平均的方式计算running_mean和running_var
        '''
        super(MyBatchnorm1d,self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(num_features).float())
        self.bias = torch.nn.Parameter(torch.zeros(num_features).float())
        #register_buffer相当于requires_grad=False的Parameter，所以两种方法都可以
        #方法一：
        self.register_buffer('running_mean',torch.zeros(num_features))
        self.register_buffer('running_var',torch.zeros(num_features))
        self.register_buffer('num_batches_tracked',torch.tensor(0))
        #方法二：
        # self.running_mean = torch.nn.Parameter(torch.zeros(num_features),requires_grad=False)
        # self.running_var = torch.nn.Parameter(torch.ones(num_features),requires_grad=False)
        # self.num_batches_tracked = torch.nn.Parameter(torch.tensor(0),requires_grad=False)
 
        self.momentum = momentum
 
    def forward(self,x):
        if self.training: #训练模型
            #数据是二维的情况下，可以这么处理，其他维的时候不是这样的，但原理都一样。
            mean_bn = x.mean(0, keepdim=True).squeeze(0) #相当于x.mean(0, keepdim=False)
            var_bn = x.var(0, keepdim=True).squeeze(0) #相当于x.var(0, keepdim=False)
 
            if self.momentum is not None:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:  #直接取平均,以下是公式变形，即 m_new = (m_old*n + new_value)/(n+1)
                self.running_mean = self.running_mean+(mean_bn.data-self.running_mean)/(self.num_batches_tracked+1)
                self.running_var = self.running_var+(var_bn.data-self.running_var)/(self.num_batches_tracked+1)
            self.num_batches_tracked += 1
        else: #eval模式
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)
 
        eps = 1e-5
        x_normalized = (x - mean_bn) / torch.sqrt(var_bn + eps)
        results = self.weight * x_normalized + self.bias
        return results


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel,self).__init__()
        linear1_features = 5
        self.linear1 = torch.nn.Linear(3,linear1_features)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(linear1_features,2)
        #设计时batchnorm放在linear1后面，所以这里用linear1的输出维度
        # self.batch_norm = torch.nn.BatchNorm1d(linear1_features,momentum=0.1)  #标准库中的Barchnorm,track_running_stats默认为True
        # self.batch_norm = torch.nn.BatchNorm1d(linear1_features,momentum=None)  #标准库中的Barchnorm,track_running_stats默认为True
        self.batch_norm = MyBatchnorm1d(linear1_features,momentum=0.1)
 
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.linear2(x)
        return x
 
#先进行一下简单训练之后，保存参数，后面的模型可以加载此函数，这样相当于给用于实验的两个模型初始化了相同的参数
train_demo = 1
if train_demo == 1:
    model = SimpleModel()
    # print(list(model.parameters()))
    # #查看模型的初始参数
    # print(model.state_dict().keys())
    # # for i, j in model.named_parameters():
    # for i,j in model.state_dict().items():
    #     print('++++',i)
    #     print('\t',j)
 
    loss_fn = torch.nn.MSELoss(size_average=False)
    optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
    model.train()
    for t,x in enumerate(dataset):
        y_pred = model(x)
        loss = loss_fn(y_pred,y)
        print(t,loss.data)
        model.zero_grad()
        loss.backward()
        optimizer.step()
 
    #查看训练后的模型参数
    print('##################The trained Model parameters###############')
    print(model.state_dict().keys())
    # for i, j in model.named_parameters():
    for i,j in model.state_dict().items():
        print('++++',i)
        print('\t',j)
    #保存模型参数
    state = {'model': model.state_dict()}
    torch.save(state,'debug_batchnorm.pth')
 

class DebugSimpleModel(torch.nn.Module):
    def __init__(self):
        super(DebugSimpleModel,self).__init__()
        linear1_features = 5
        self.linear1 = torch.nn.Linear(3,linear1_features)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(linear1_features,2)
        self.batch_norm = MyBatchnorm1d(linear1_features,momentum=0.9)  #使用自定义的Batchnorm
        # self.batch_norm = MyBatchnorm1d(linear1_features,momentum=None)  #使用自定义的Batchnorm
 
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.linear2(x)
        return x
 
 
#查看训练后的模型参数
print('##################The trained Model parameters###############')
model_param_dict = torch.load('debug_batchnorm.pth')['model']
print(model_param_dict.keys())
# for i, j in model.named_parameters():
for i,j in model_param_dict.items():
    print('++++',i)
    print('\t',j)
 
'''
以下的过程中都是在train模式下进行的，并且没有进行损失计算和梯度更新，
但这个过程中running_mean和running_var会进行更新，所以也验证了
running_mean和running_var只受模型的模式（train模型或eval模型)的影响，
与是否进行反向传播(loss.backward)和梯度更新(optimiter.step)没有关系。
实验一：
1. 标准库函数的参数设置为 torch.nn.BatchNorm1d(linear1_features,momentum=0.1)
2. 自定义函数的参数设置为 MyBatchnorm1d(linear1_features,momentum=0.9)
3. 相同的输入，对比输出的参数值是否相同
实验二：
1. 标准库函数的参数设置为 torch.nn.BatchNorm1d(linear1_features,momentum=None)
2. 自定义函数的参数设置为 MyBatchnorm1d(linear1_features,momentum=None)
3. 相同的输入，对比输出的参数值是否相同
'''
test_demo = 1
if test_demo == 1:
    test_model = SimpleModel()
    test_model.load_state_dict(torch.load('debug_batchnorm.pth')['model'])
    test_model.train()
    for t,x in enumerate(dataset):
        y_pred = test_model(x)
    print('\n++++++++++  Norm output  ++++++++++++++++')
    for i,j in test_model.state_dict().items():
        print('++++',i)
        print('\t',j)
 
debug_demo = 1
if debug_demo == 1:
    debug_model = DebugSimpleModel()
    #因为自定义的模型参数与标准模型的参数完全一样，所以把标准模型作为预训练的模型（即可以加载标准模型的训练后的参数作为自己的参数）
    debug_model.load_state_dict(torch.load('debug_batchnorm.pth')['model'])
    debug_model.train()
    for t,x in enumerate(dataset):
        y_pred = debug_model(x)
 
    print('\n++++++++++++ Mymodel Output ++++++++++++++')
    for i,j in debug_model.state_dict().items():
        print('++++',i)
        print('\t',j)