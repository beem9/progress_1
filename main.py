import matplotlib.pyplot as plt
import networkx as nx

# was using main to try each code, part c and part b have their own .py file
n, m = 8, 8
G = nx.grid_2d_graph(n, m)


pos = {(x, y): (y, -x) for x, y in G.nodes()}


plt.figure(figsize=(8, 8))
nx.draw(G, pos, node_color='lightgreen', edge_color='black',
        with_labels=True, labels={node: node for node in G.nodes()})
plt.title('8x8 Grid Network')
plt.show()

