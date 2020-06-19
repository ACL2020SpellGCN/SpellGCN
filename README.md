#SpellGCN
========

SpellGCN is a chinese spelling checki method to embed visual and phological knowledge into BERT.
This repository contains data, evaluation and training scripts.

Citation:

```
@inproceedings{DBLP:journals/corr/abs-2004-14166,
  author    = {Xingyi Cheng and
               Weidi Xu and
               Kunlong Chen and
               Shaohua Jiang and
               Feng Wang and
               Taifeng Wang and
               Wei Chu and
               Yuan Qi},
  title={SpellGCN: Incorporating Phonological and Visual Similarities into
               Language Models for Chinese Spelling Check},
  booktitle={ACL},
  year={2020}
}

```
This is the official code for paper titled "SpellGCN: Incorporating Phonological and Visual Similarities into Chinese Spelling Check". Accepted by ACL2020

## How to run the scripts reproduce our results

The code is based on Tensorflow==1.13.1 and python 2.7 or higher

To replicate the results from the paper, run commands as follows:

```
cd scripts/
conda create -n spellgcn python=2.7.1
source activate spellgcn
pip intall tensorflow==1.13.1
sh run.sh
```

Since SpellGCN is based on BERT, the path to the BERT directory should be provided in the run.sh.
The default training data are merged by SIGHAN13, SIGHAN14, SIGHAN15.




