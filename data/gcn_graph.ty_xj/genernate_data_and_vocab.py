#!/usr/bin/env python
# encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding("utf8")
graphs = {}
vocab = {}
for line in open("./vocab.txt"):
    vocab.setdefault(line.strip(), 1)

for i, a in enumerate(open("./SimilarShape_simplied.txt")):
    word, simshape =  a.split(",")
    if word.strip() == "𫓧":
        continue
    if word.strip() not in vocab:
        continue
    for _simshape in simshape.decode("utf8"):
        _simshape = _simshape.strip().encode("utf8")
        if _simshape not in vocab:
            continue
        if _simshape != "" and _simshape != u"𫓧":
            graphs.setdefault((word,_simshape),[]).append( "形近")

for i, a in enumerate(open("./SimilarPronunciation_simplied.txt")):
    if i == 0:
        continue
    word, swst, swdt, simwst, simwdt, sbs =  a.split("\t")
    if word.strip() == "𫓧":
        continue
    if word.strip() not in vocab:
        continue
    for _swst in swst.decode("utf8"):
        _swst = _swst.strip().encode("utf8")
        if _swst not in vocab:
            continue
        if _swst != "" and _swst != u"𫓧":
            graphs.setdefault((word,_swst),[]).append("同音同调")
    for _swdt in swdt.decode("utf8"):
        _swdt = _swdt.strip().encode("utf8")
        if _swdt not in vocab:
            continue
        if _swdt != "" and _swdt != u"𫓧":
            graphs.setdefault((word,_swdt), []).append("同音异调")
    for _simwst in simwst.decode("utf8"):
        _simwst = _simwst.strip().encode("utf8")
        if _simwst not in vocab:
            continue
        if _simwst != "" and _simwst != u"𫓧":
            graphs.setdefault((word,_simwst), []).append("近音同调")
    for _simwdt in simwdt.decode("utf8"):
        _simwdt = _simwdt.strip().encode("utf8")
        if _simwdt not in vocab:
            continue
        if _simwdt != "" and _simwdt != u"𫓧":
            graphs.setdefault((word,_simwdt),[]).append("近音异调")
    for _sbs in sbs.decode("utf8"):
        _sbs = _sbs.strip().encode("utf8")
        if _sbs not in vocab:
            continue
        if _sbs != "" and _sbs != u"𫓧":
            graphs.setdefault((word,_sbs),[]).append("同部首同笔画")


for (e1,e2), relations in graphs.items():
    for relation in relations:
      print("%s|%s|%s" %(e1, e2, relation))

