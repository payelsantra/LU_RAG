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

import json
evidence_text={}
with open('/home/user/data/corpus/nei/NEI_bucket.jsonl','r',encoding='utf-8') as f:
    for idx,line in enumerate(f):
        json_obj=json.loads(line.strip())
        evidence_text[json_obj['id']]=json_obj['contents']

import pandas as pd

test_data = pd.read_csv('/media/user/Expansion/phd_new/my_desktop/payel/Fact_verification/emnlp_extension/fever/data/shared_task_dev_fever_data.csv')

# id_claim_dict=dict(zip(list(test_data['id']),list(test_data['claim'])))
# id_label_dict=dict(zip(list(test_data['id']),list(test_data['label'])))
test_data_claim=dict(zip(test_data['id'],test_data['claim']))
test_data_label=dict(zip(test_data['id'],test_data['label']))


results=[]
top_k=50
for index, row in test_data.iterrows():
    query_id = row['id']
    query = row['claim']
    hits = hybrid_searcher.search(query,k=top_k)

    # Store results for each query
    for i in range(len(hits)):  # Limit to top 10 results
        results.append({
            'query_id': query_id,
            'query': query,
            'docid': hits[i].docid,
            'score': hits[i].score
        })
        
from collections import defaultdict
query_dict = defaultdict(list)

for entry in results:
    query_dict[entry["query_id"]].append(entry["docid"])
query_dict = dict(query_dict)
```
Post-processing is as follows:
```
from tqdm import tqdm
result_dict={}
for i in tqdm(query_dict):
    query_id=i
    for k,l in enumerate(query_dict[i]):
        key=str(i)+"_"+str(k+1)
        doc_id=l
        evid=evidence_text[int(doc_id)].strip()
        if query_id in result_dict:
            result_dict[query_id][key] = [evid, doc_id,'NEI']
        else:
            result_dict[query_id]={}
            result_dict[query_id][key]=[evid,doc_id,'NEI']

import pickle
fl_p=open("/home/user/result/bm25_ret/contriever/bm25_contriever_results.pickle","wb")
pickle.dump(result_dict,fl_p)
fl_p.close()
```
