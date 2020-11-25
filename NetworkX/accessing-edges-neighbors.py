# from: https://networkx.org/documentation/latest/tutorial.html

import networkx as nx


# Accessing edges and neighbors
G = nx.Graph([(1, 2, {"color": "yellow"})])
G[1]  # same as G.adj[1] -># AtlasView({2: {'color': 'yellow'}})
G[1][2] #-> {'color': 'yellow'}
G.edges[1, 2] #-> {'color': 'yellow'}

G.add_edge(1, 3)
G[1][3]['color'] = "blue"
G.edges[1, 2]['color'] = "red"
G.edges[1, 2] #-> {'color': 'red'}

# Fast examination of all (node, adjacency) pairs is achieved using G.adjacency(), or G.adj.items().
FG = nx.Graph()
FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
for n, nbrs in FG.adj.items():
   for nbr, eattr in nbrs.items():
       wt = eattr['weight']
       if wt < 0.5: print(f"({n}, {nbr}, {wt:.3})")
# or Convenient access to all edges is achieved with the edges property.
for (u, v, wt) in FG.edges.data('weight'):
    if wt < 0.5:
        print(f"({u}, {v}, {wt:.3})")
# 使用邻接矩阵去判断是比使用边的2倍，
# 例如(1,2)这个边满足条件，使用边判断一次即可，如果使用邻接矩阵，当分别遍历到1,2两个节点时，都会判断到这条边。


# Adding attributes to graphs, nodes, and edges
# Graph attributes:
# Assign graph attributes when creating a new graph
G = nx.Graph(day="Friday")
G.graph # -> {'day': 'Friday'}

# modify attributes later
G.graph['day'] = "Monday"
G.graph #-> {'day': 'Monday'}

# Node attributes:
# Add node attributes using add_node(), add_nodes_from(), or G.nodes
G.add_node(1, time='5pm')
G.add_nodes_from([3], time='2pm')
G.nodes[1] #-> {'time': '5pm'}
G.nodes[1]['room'] = 714
G.nodes.data() #-> NodeDataView({1: {'time': '5pm', 'room': 714}, 3: {'time': '2pm'}})

# Edge Attributes
# Add/change edge attributes using add_edge(), add_edges_from(), or subscript notation.
G.add_edge(1, 2, weight=4.7 )
G.add_edges_from([(3, 4), (4, 5)], color='red')
G.add_edges_from([(1, 2, {'color': 'blue'}), (2, 3, {'weight': 8})])
G[1][2]['weight'] = 4.7
G.edges[3, 4]['weight'] = 4.2


print('testing')