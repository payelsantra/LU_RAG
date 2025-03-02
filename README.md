# LU_RAG

This is the repository for the paper: `Analyzing the Role of Retrieval Models in Labeled and Unlabeled Context Selection for Fact Verification`.

This repository allows the replication of all results reported in the papers. In particular, it is organized as follows:
- [Prerequisites](#Prerequisites)
- [Data Preparation](#Data-Preparation)
  - [Data Preprocessing](#Data-Preprocessing)
  - [Corpus Preprocessing](#Corpus-Preprocessing)
- [How do I search?](#search)
  - [Single-stage Ranker](#Single-stage)
  - [Two-stage Ranker](#Two-stage)
- [Post-processing](#post-processing)
-  [Replicating Results](#Replicating-Results)
    - [L-RAG](#L-RAG)
    - [U-RAG](#U-RAG)
    - [LU-RAG](#LU-RAG)

## Prerequisites
We recommend running all the things in a Linux environment. 
Please create a conda environment with all required packages, and activate the environment by the following commands:
```
$ conda create -n pyserini_env python==3.10
$ conda activate pyserini_env
```
We have used [pyserini](https://github.com/castorini/pyserini) for each search. 

Install via PyPI:
```
$ pip install pyserini
```
Pyserini is built on Python 3.10 (other versions might work, but YMMV) and Java 21 (due to its dependency on [Anserini](https://github.com/castorini/anserini)).

## Data Preparation
### Data Preprocessing

### Corpus Preprocessing

## How do I search?
### Single-stage Ranker
For this paper, we uniformly retrieved the top 50 candidates.
#### BM25
```
## indexing
!python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /home/user/data/corpus/nei/ \
  --index /home/user/data/index_file/nei/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw

## search
!python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index /home/user/data/index_file/nei/ \
  --topics /home/user/data/data.tsv \
  --output /home/user/result/bm25_ret/bm25_nei_ret.txt \
  --bm25 --k1 0.9 --b 0.4 --hits 100
```

#### Contriever-E2E

#### ColBERT-E2E
### Two-stage Ranker
