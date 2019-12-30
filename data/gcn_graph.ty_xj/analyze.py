#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding("utf8")
from collections import defaultdict

def load_relation_vocab(relation_vocab_fn):
  relation_vocab = {}
  for line in open(relation_vocab_fn):
    relation_vocab[line.split(u" ")[0].strip()] = line.split(u" ")[1].strip()
  return relation_vocab

def load_confusionset(confusionset_fn):
  pairs = defaultdict(list)
  with open(confusionset_fn) as reader:
    for line in reader:
      pairs[line.split(u"|")[2].strip()].append(line.split(u"|")[0:2])
  return pairs


confusionset = load_confusionset("spellGraphs.txt")
relation_vocab = load_relation_vocab("relation_vocab.txt")

print(relation_vocab)
print(confusionset.keys())

mapped_confusionset = defaultdict(list)
for k in confusionset:
  if k in relation_vocab:
    mapped_confusionset[relation_vocab[k]].extend(confusionset[k])

for k in mapped_confusionset:
  chars = []
  pairs = []
  for pair in mapped_confusionset[k]:
    chars.extend(pair)
    pairs.append(u"|".join(pair))
  print(k, len(set(chars)), len(set(pairs)))


