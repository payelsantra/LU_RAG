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
Convert your csv file into a tsv file `/home/user/data/data.tsv`.
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
```
##indexing
!python -m pyserini.encode \
  input \
    --corpus /home/user/data/corpus/nei/NEI_bucket.jsonl \
    --fields text \
    --delimiter "\n" \
    --shard-id 0 \
    --shard-num 1 \
  output \
    --embeddings /home/user/data/dense_retrieval/indom/embedding_nei_dense1 \
    --to-faiss \
  encoder \
    --encoder facebook/contriever-msmarco \
    --fields text \
    --batch 32 \
    --fp16

#search
!python -m pyserini.search.faiss \
  --index /home/user/data/dense_retrieval/indom/embedding_nei_dense1 \
  --topics /home/user/data/data.tsv \
  --encoder facebook/contriever-msmarco \
  --output /home/user/result/contriever_ret/nei_contriever_results.txt \
  --batch-size 64 \
  --threads 4 \
  --hits 50
```
#### ColBERT-E2E
```
##indexing
!python -m pyserini.encode \
  input \
    --corpus /home/user/data/corpus/nei/NEI_bucket.jsonl \
    --fields text \
    --delimiter "\n" \
    --shard-id 0 \
    --shard-num 1 \
  output \
    --embeddings /home/user/data/dense_retrieval/indom/embedding_nei_dense1 \
    --to-faiss \
  encoder \
    --encoder castorini/tct_colbert-v2-hnp-msmarco \
    --fields text \
    --batch 32 \
    --fp16

#search
!python -m pyserini.search.faiss \
  --index /home/user/data/dense_retrieval/indom/embedding_nei_dense1 \
  --topics /home/user/data/data.tsv \
  --encoder castorini/tct_colbert-v2-hnp-msmarco \
  --output /home/user/result/contriever_ret/nei_colbert_results.txt \
  --batch-size 64 \
  --threads 4 \
  --hits 50
```
### Two-stage Ranker
In our retrieve-and-rerank approach, we first retrieve the top 100 candidate documents using the same configurations as our single-stage retriever, and then re-ranks (also configured identically as explained in single-stage retriever) to narrow these down to the top 50 documents.

#### BM25»Contriever
```
from pyserini.search.faiss import FaissSearcher
from pyserini.encode import AutoQueryEncoder

from pyserini.encode import TctColBertQueryEncoder
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher
from pyserini.search.hybrid import HybridSearcher

sparse_searcher = LuceneSearcher('/home/user/data/index_file/nei/')
encoder = TctColBertQueryEncoder('facebook/contriever-msmarco')
dense_searcher = FaissSearcher('/home/user/data/dense_retrieval/indom/embedding_nei_dense1', encoder)
hybrid_searcher = HybridSearcher(dense_searcher, sparse_searcher)
```
