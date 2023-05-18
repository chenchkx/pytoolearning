
import torch
from torch.nn import Linear, ReLU
from torch_geometric.nn import HeteroConv

from torch_geometric.data import Data, Batch

# Define node features for each node type
node_type1 = torch.randn(5, 32)
node_type2 = torch.randn(10, 32)

# Define edge indices for each edge type
edge_type1_indices = torch.tensor([[0, 1, 2, 3], [2, 3, 4, 1]])
edge_type2_indices = torch.tensor([[0, 1, 2, 2], [2, 3, 4, 1]])

# Concatenate node features and indices into a PyTorch Geometric Data object
data = Data(
    node_type1=node_type1,
    node_type2=node_type2,
    edge_index_etype1=edge_type1_indices,
    edge_index_etype2=edge_type2_indices
)

# Add a global edge connecting all node types
global_edge_index = torch.tensor([[i for i in range(data.num_nodes)], [i for i in range(data.num_nodes)]])
data.edge_index_global = global_edge_index.transpose(0, 1)

# Define target labels
target = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])

# Construct a batch of a single graph
batch = Batch.from_data_list([data])
batch.y = target



class HeteroGNN(torch.nn.Module):
    def __init__(self, node_input_dim, out_dim):
        super(HeteroGNN, self).__init__()

        self.node_input_dim = node_input_dim
        self.out_dim = out_dim

        # Define HeteroConv layers for each edge type
        self.conv1 = HeteroConv({
            'node_type1': Linear(node_input_dim['node_type1'], out_dim),
            'node_type2': Linear(node_input_dim['node_type2'], out_dim),
            'edge_type1': Linear(node_input_dim['node_type1'], out_dim),
            'edge_type2': Linear(node_input_dim['node_type2'], out_dim),
            'global': Linear(1, out_dim) 
        })

        self.conv2 = HeteroConv({
            'node_type1': Linear(out_dim, out_dim),
            'node_type2': Linear(out_dim, out_dim),
            'edge_type1': Linear(out_dim, out_dim),
            'edge_type2': Linear(out_dim, out_dim),
            'global': Linear(out_dim, out_dim)
        })

        self.conv3 = HeteroConv({
            'node_type1': Linear(out_dim, out_dim),
            'node_type2': Linear(out_dim, out_dim),
            'edge_type1': Linear(out_dim, out_dim),
            'edge_type2': Linear(out_dim, out_dim),
            'global': Linear(out_dim, out_dim)
        })

        self.conv4 = HeteroConv({
            'node_type1': Linear(out_dim, out_dim),
            'node_type2': Linear(out_dim, out_dim),
            'edge_type1': Linear(out_dim, out_dim),
            'edge_type2': Linear(out_dim, out_dim),
            'global': Linear(out_dim, out_dim)
        })

        self.relu = ReLU()

    def forward(self, x, edge_index, edge_types):
        # x: a dictionary of tensors for each node type
        # edge_index: a list of edge indices for each edge type, i.e., [(src, dst), (src, dst), ...]
        # edge_types: a list of edge types corresponding to the edge_index list

        x = self.conv1(x, edge_index, edge_types)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_types)
        x = self.relu(x)
        x = self.conv3(x, edge_index, edge_types)
        x = self.relu(x)
        x = self.conv4(x, edge_index, edge_types)

        return x

# Initialize the Hetero Graph Neural Network model
model = HeteroGNN(node_input_dim = {'node_type1': 32, 'node_type2': 32}, out_dim = 64)

import torch.optim as optim
from torch.nn.functional import cross_entropy

optimizer = optim.SGD(model.parameters(), lr=0.1)

# Train the model for 10 epochs
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    out = model(batch)
    loss = cross_entropy(out, batch.y)
    loss.backward()
    optimizer.step()
    print('Epoch:', epoch, ', Loss:', loss.item())

# Test the model
model.eval()
out = model(batch)
pred = torch.argmax(out, dim=1)
print("Predictions:", pred)
print("Actual:", batch.y)
