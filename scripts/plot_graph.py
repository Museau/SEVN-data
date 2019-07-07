import os
from shutil import copyfile
import networkx as nx
import matplotlib.pyplot as plt
import subprocess
import argparse
import pandas as pd
import collections

parser = argparse.ArgumentParser("Creates panoramas from vuze pictures")
parser.add_argument('--input_path', type=str, help='Input path')
parser.add_argument('--zones', action='store_true', help='Different color for streets and corners')
args = parser.parse_args()

print("Path : " + args.input_path)
input_path = args.input_path
zones = args.zones

G = nx.read_gpickle(input_path + '/graph.pkl')

pos = {k: v.get("coords")[0:2] for k, v in G.nodes(data=True)}
if zones:
    meta_df = pd.read_hdf(input_path + "/meta.hdf5", key="df", index=False)
    corners = [node for node in G.nodes if node in meta_df[meta_df.type == 'intersection'].frame]
    streets = [node for node in G.nodes if node in meta_df[meta_df.type == 'street_segment'].frame]
    other = [node for node in G.nodes if node not in meta_df[meta_df.type == 'street_segment'].frame]
    other = [node for node in other if node not in meta_df[meta_df.type == 'intersection'].frame]

    nx.draw_networkx_nodes(G, pos,
                           nodelist=corners,
                           node_color='#e23f3a',
                           node_size=1,
                           alpha=0.8,
                           with_label=True)

    nx.draw_networkx_nodes(G, pos,
                           nodelist=streets,
                           node_color='#000cda',
                           node_size=1,
                           alpha=0.8)

    nx.draw_networkx_nodes(G, pos,
                           nodelist=other,
                           node_color='g',
                           node_size=20,
                           alpha=0.8)
    edges = nx.draw_networkx_edges(G, pos=pos)

else:
    nx.draw(G, pos,node_color='r', node_size=1)

plt.axis('equal')
plt.savefig('graph.png', transparent=True, dpi=1000)
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
