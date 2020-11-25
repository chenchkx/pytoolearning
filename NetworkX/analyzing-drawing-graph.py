# from: https://networkx.org/documentation/latest/tutorial.html

import networkx as nx


# analyzing graphs
# The structure of G can be analyzed using various graph-theoretic functions such as:
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3)])
G.add_node("spam")       # adds node "spam"
G.add_edge(1,"spam")
list(nx.connected_components(G)) #-> [{1, 2, 3}, {'spam'}]
sorted(d for n, d in G.degree()) #-> [0, 1, 1, 2]
nx.clustering(G) #-> {1: 0, 2: 0, 3: 0, 'spam': 0}
sp = dict(nx.all_pairs_shortest_path(G))
sp[3] #->{3: [3], 1: [3, 1], 2: [3, 1, 2]}


# drawing graphing
import matplotlib.pyplot as plt


G = nx.petersen_graph()
plt.subplot(121)
# <matplotlib.axes._subplots.AxesSubplot object at ...>
nx.draw(G, with_labels=True, font_weight='bold') # 直接draw 是 不需要Pos 信息的
plt.subplot(122)
# <matplotlib.axes._subplots.AxesSubplot object at ...>
pos=nx.circular_layout(G)
nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
# nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
plt.show()

print('testing')