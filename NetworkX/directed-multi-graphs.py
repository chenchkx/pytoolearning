# from: https://networkx.org/documentation/latest/tutorial.html

import networkx as nx


# Directed graphs
# The DiGraph class provides additional methods and properties specific to directed edges,
# e.g., DiGraph.out_edges, DiGraph.in_degree, DiGraph.predecessors(), DiGraph.successors() etc.
DG = nx.DiGraph()
DG.add_weighted_edges_from([(1, 2, 0.5), (3, 1, 0.75)])
DG.out_degree(1, weight='weight') #-> 0.5
DG.degree(1, weight='weight') #-> 1.25
list(DG.successors(1)) #-> [2]
list(DG.neighbors(1)) #-> [2]


# Multigraphs
# NetworkX provides classes for graphs which allow multiple edges between any pair of nodes.
# The MultiGraph and MultiDiGraph classes allow you to add the same edge twice, possibly with different edge data.
MG = nx.MultiGraph()
MG.add_weighted_edges_from([(1, 2, 0.5), (1, 2, 0.75), (2, 3, 0.5)])
dict(MG.degree(weight='weight')) #-> {1: 1.25, 2: 1.75, 3: 0.5}
GG = nx.Graph()
for n, nbrs in MG.adjacency():
   for nbr, edict in nbrs.items():
       minvalue = min([d['weight'] for d in edict.values()])
       GG.add_edge(n, nbr, weight = minvalue)

nx.shortest_path(GG, 1, 3) #-> [1, 2, 3]

print('testing')








