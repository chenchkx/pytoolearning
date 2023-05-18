import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv

# 定义一个简单的异构图
edge_index_dict = {'user_item': torch.tensor([[0, 1], [1, 0]]),
                   'item_user': torch.tensor([[1, 0], [0, 1]])}
x_dict = {'user': torch.randn(2, 16),
          'item': torch.randn(2, 8)}
user_price = torch.tensor([1.99, 4.99]).view(-1, 1)
item_gender = torch.tensor([0, 1]).view(-1, 1)
x_dict['item'] = torch.cat([x_dict['item'], user_price], dim=-1)
x_dict['user'] = torch.cat([x_dict['user'], item_gender], dim=-1)

# 定义一个包含HeteroConv的神经网络模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = HeteroConv({
            'user': 16,
            'item': 9,
        }, out_channels=32)
        self.lin = torch.nn.Linear(32, 1)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv(x_dict, edge_index_dict)
        x = x_dict['user'] + x_dict['item']
        x = F.relu(x)
        x = self.lin(x)
        return x

# 创建示例输入数据
x_dict = {'user': torch.randn(2, 16),
          'item': torch.randn(2, 8)}
edge_index_dict = {'user_item': torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])}

model = Net()
output = model(x_dict, edge_index_dict)