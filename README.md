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
  - [For Single-stage](#forSingle-stage)
  - [For Two-stage](#forTwo-stage)
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
Pyserini is built on Python 3.10 (other versions might work, but YMMV) and Java 21 (due to its dependency on [Anserini](https://github.com/castorini/anserini)). For all the retrievers we need to install Pyserini of Version: 0.44.0, only for bm25>>monot5 we need to install Pyserini of Version: 0.16.1.

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

test_data = pd.read_csv('/home/user/data/data.csv')
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

#### BM25»ColBERT
```
from pyserini.search.faiss import FaissSearcher
from pyserini.encode import AutoQueryEncoder

from pyserini.encode import TctColBertQueryEncoder
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher
from pyserini.search.hybrid import HybridSearcher

sparse_searcher = LuceneSearcher('/home/user/data/index_file/nei/')
encoder = TctColBertQueryEncoder('castorini/tct_colbert-msmarco')
dense_searcher = FaissSearcher('/home/user/data/dense_retrieval/indom/embedding_nei_dense1', encoder)
hybrid_searcher = HybridSearcher(dense_searcher, sparse_searcher)

import json
evidence_text={}
with open('/home/user/data/corpus/nei/NEI_bucket.jsonl','r',encoding='utf-8') as f:
    for idx,line in enumerate(f):
        json_obj=json.loads(line.strip())
        evidence_text[json_obj['id']]=json_obj['contents']

import pandas as pd

test_data = pd.read_csv(' /home/user/data/data.csv')
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
#### BM25»MonoT5
For this, you need to install Pyserini of Version: 0.16.1.
```
## indexing
from pyserini.search.lucene import LuceneSearcher
from pygaggle.rerank.transformer import MonoT5
from pygaggle.rerank.base import Reranker, Query, Text
import pandas as pd

!python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /home/user/data/corpus/nei/ \
  --index /home/user/data/index_file/nei_16/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw

# Load the test data
test_data = pd.read_csv('/home/user/data/data.csv')
id_claim_dict = dict(zip(list(test_data['id']), list(test_data['claim'])))

# Step 1: BM25 Retrieval
bm25_searcher = LuceneSearcher('/home/user/data/index_file/nei_16/')
bm25_searcher.set_bm25(k1=0.9, b=0.4)

# Step 2: Initialize monoT5 reranker
reranker = MonoT5('castorini/monot5-base-msmarco')

reranked_results = {}

for query_id, query_text in id_claim_dict.items():
    hits = bm25_searcher.search(query_text, k=100)  # Retrieve top 100 with BM25
    query = Query(query_text)
    texts = [Text(bm25_searcher.doc(hit.docid).raw(), {'docid': hit.docid}, 0) for hit in hits]
    reranked = reranker.rerank(query, texts)
    reranked = sorted(reranked, key=lambda x: x.score, reverse=True)  # Sort by score
    reranked_results[query_id] = [(text.metadata['docid'], text.score) for text in reranked[:50]]
```

## Post-processing
### For Single-stage
Download the ```train.jsonl``` file from https://fever.ai/dataset/fever.html. (As we have used Fever training data as source data)
```
import json
with open("/data/train.jsonl", 'r') as json_file:
    json_list = list(json_file)
whole_dict={}
for json_str in json_list:
    result = json.loads(json_str)
    id_nw=result['id']
    label=result['label']
    claim=result['claim']
    whole_dict[id_nw]={"claim":claim}
    whole_dict[id_nw].update({"label":label})

def make_dict(give_dict,token):
  updated_list={}
  for i in give_dict:
    token_element=give_dict[i][token]
    updated_list[i]=token_element
  return updated_list
tr_claim_data_dict=make_dict(whole_dict,token="claim")

result_bm25_path = "/home/user/result/bm25_ret/bm25_nei_ret.txt"
result_dict={}
with open(result_bm25_path, "r") as file:
    for line in file:
        parts=line.strip().split()
        query_id = int(parts[0])
        doc_id = tr_claim_data_dict[int(parts[2])].strip()  # Remove the "doc" prefix
        score = float(parts[4])
        evi_order=parts[3]
        key = f"{query_id}_{evi_order}"
        if query_id in result_dict:
            result_dict[query_id][key] = [doc_id, int(parts[2]),'NEI']
        else:
            result_dict[query_id]={}
            result_dict[query_id][key]=[doc_id,int(parts[2]),'NEI']

import pickle
fl_p=open("/home/user/result/bm25_test_ret.pickle","wb")
pickle.dump(result_dict,fl_p)
fl_p.close()
```

### For Two-stage
The post processing for BM25>>Contriever and BM25>>ColBERT are as follows:
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
fl_p=open("/home/user/result/bm25_ret/dense/bm25_dense_results.pickle","wb")
pickle.dump(result_dict,fl_p)
fl_p.close()
```
The post-processing for BM25>>MonoT5 is as follows:
```
from tqdm import tqdm
import json
evidence_text={}
with open('/home/user/data/corpus/nei/NEI_bucket.jsonl','r',encoding='utf-8') as f:
    for idx,line in enumerate(f):
        json_obj=json.loads(line.strip())
        evidence_text[json_obj['id']]=json_obj['contents']

result_dict={}
for i in tqdm(reranked_results):
    query_id=i
    for k,l in enumerate(reranked_results[i]):
        key=str(i)+"_"+str(k+1)
        doc_id=l[0]
        evid=evidence_text[int(doc_id)].strip()
        if query_id in result_dict:
            result_dict[query_id][key] = [evid, doc_id,'NEI']
        else:
            result_dict[query_id]={}
            result_dict[query_id][key]=[evid,doc_id,'NEI']

import pickle
fl_p=open("/home/user/result/bm25_ret/dense/bm25_monot5_results.pickle","wb")
pickle.dump(result_dict,fl_p)
fl_p.close()
```
