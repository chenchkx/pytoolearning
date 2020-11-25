# from: https://networkx.org/documentation/latest/tutorial.html

import networkx as nx

# Create an empty graph with no nodes and no edges.
G = nx.Graph()

# add nodes
G.add_node(11)
G.add_nodes_from([12, 13])
# Nodes from one graph can be incorporated into another:
H = nx.path_graph(10)
H = nx.path_graph(10)
G.add_nodes_from(H)


# add edges
G.add_edge(1, 2)
e = (2, 3)
G.add_edge(*e)  # unpack edge tuple*
# by adding a list of edges,
G.add_edges_from([(1,2),(1,3)])


# remove all nodes and edges
G.clear()


# we add new nodes/edges and NetworkX quietly ignores any that are already present.
G.add_edges_from([(1, 2), (1, 3)])
G.add_node(1)
G.add_edge(1, 2)
G.add_node("spam")        # adds node "spam"
G.add_nodes_from("spam")  # adds 4 nodes: 's', 'p', 'a', 'm'
G.add_edge(3, 'm')
# At this stage the graph G consists of 8 nodes and 3 edges, as can be seen by:
G.number_of_nodes()
G.number_of_edges()


G = nx.DiGraph()
G.add_edge(2, 1)   # adds the nodes in order 2, 1
G.add_edge(1, 3)
G.add_edge(2, 4)
G.add_edge(1, 2)
assert list(G.successors(2)) == [1, 4]
# 找子节点: DiGraph.successors(n) Return a list of successor nodes of n.
# 找父节点：DiGraph.predecessors(n) Return a list of predecessor nodes of n.
# 求距离某一节点（0）的最短路径： nx.shortest_path_length(G, 0, node)
assert list(G.edges) == [(2, 1), (2, 4), (1, 3), (1, 2)]



print('testing')




