# from: https://networkx.org/documentation/latest/tutorial.html

import networkx as nx

G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3)])
G.add_node(1)
G.add_edge(1, 2)
G.add_node("spam")        # adds node "spam"
G.add_nodes_from("spam")  # adds 4 nodes: 's', 'p', 'a', 'm'
G.add_edge(3, 'm')


# Examining elements of a graph
# The basic graph properties facilitate reporting: G.nodes, G.edges, G.adj and G.degree.
list(G.nodes) #-> [1, 2, 3, 'spam', 's', 'p', 'a', 'm']
list(G.edges) #-> [(1, 2), (1, 3), (3, 'm')]
list(G.adj[1]) #-> [2, 3] # or list(G[1]), or list(G.neighbors(1))
G.degree[1] #-> 2
G.edges([2, 'm']) #-> EdgeDataView([(2, 1), ('m', 3)])
G.degree([2, 3]) #-> DegreeView({2: 1, 3: 2})


# Removing elements from a graph
G.remove_node(2)
G.remove_nodes_from("spam")
list(G.nodes)
# [1, 3, 'spam']
G.remove_edge(1, 3)


# Using the graph constructors
# Graph objects do not have to be built up incrementally - data specifying graph structure can be passed directly to the
# constructors of the various graph classes. When creating a graph structure by instantiating one of the graph classes
# you can specify data in several formats.
G.add_edge(1, 2)
H = nx.DiGraph(G)   # create a DiGraph using the connections from G
list(H.edges()) #-> [(1, 2), (2, 1)]
edgelist = [(0, 1), (1, 2), (2, 3)]
H = nx.Graph(edgelist)


# What to use as nodes and edges
# consider using convert_node_labels_to_integers() to obtain a more traditional graph with integer labels.


print('testing')