"""Stitch panoramas"""

import os
from shutil import copyfile
import networkx as nx
import matplotlib.pyplot as plt
import subprocess
import argparse
import collections

parser = argparse.ArgumentParser("Creates panoramas from vuze pictures")
parser.add_argument('--input_path', type=str, help='graph path')
args = parser.parse_args()

print("Path : " + args.input_path)
input_path = args.input_path

G = nx.read_gpickle(input_path)
pos = {k: v.get("coords")[0:2] for k, v in G.nodes(data=True)}

# nx.draw_networkx_nodes(G, pos,
#                        nodelist=G.nodes,
#                        node_color='r',
#                        node_size=10,
#                        alpha=0.8,
#                        with_label=True)

# # First node is Green
# nx.draw_networkx_nodes(G, pos,
#                        nodelist={15},
#                        node_color='g',
#                        node_size=50,
#                        alpha=0.8)

# # Second node is blue
# nx.draw_networkx_nodes(G, pos,
#                        nodelist={16},
#                        node_color='b',
#                        node_size=50,
#                        alpha=0.8)
# edges = nx.draw_networkx_edges(G, pos=pos)

# nx.draw_networkx(G, pos,
#                  nodelist=G.nodes,
#                  node_color='r',
#                  node_size=10,
#                  alpha=0.8,
#                  with_label=True)
nx.draw(G, pos,node_color='r', node_size=1)
plt.axis('equal')
plt.show()

print("num_edges: "+ str(len(G.edges)))
print("num_nodes: "+ str(len(G.nodes)))

degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
# print "Degree sequence", degree_sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color='b')

plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xticks([d + 0.4 for d in deg])
ax.set_xticklabels(deg)
plt.show()
