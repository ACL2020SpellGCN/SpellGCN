#!/usr/bin/env python
# encoding: utf-8

import matplotlib.pyplot as plt
import networkx as nx
import sys
import numpy as np
reload(sys)
sys.setdefaultencoding("utf8")


G = nx.Graph()

vocab = {}
rel_vocab = {}
k = 0
relk = 0
g_data = []

for line in open("./spellGraphs.txt"):
    if np.random.rand() > 0.0001:
      continue
    e1, e2, rel =line.strip().split("|")
    if e1 not in vocab:
      vocab.setdefault(e1, k)
      k += 1
    if e2 not in vocab:
      vocab.setdefault(e2, k)
      k += 1
    if rel not in rel_vocab:
      rel_vocab.setdefault(rel, relk)
      relk += 1
    #g_data.append((vocab[e1], vocab[e2], rel_vocab[rel]))
    G.add_edge(vocab[e1], vocab[e2], weight=rel_vocab[rel])

nx.draw(G)
plt.show()

