# DeepQLM-ECIR2021
## Github repo for ECIR2021 short paper: "Deep Query Likelihood Model for Information Retrieval"

We rely on Huggingface [transformers](https://huggingface.co/transformers/) library and [docTquery-T5](https://github.com/castorini/docTTTTTquery) to implement T5 deep query likelihood model.

1. Download MS MARCO passage collection from https://github.com/microsoft/MSMARCO-Passage-Ranking, unzip and put `collection.tsv` in `data/`.
2. Download `t5-base.zip` from https://github.com/castorini/docTTTTTquery, unzip and rename the folder to `t5-base-tf`.
3. Since MS MARCO dev set has many quries, in this repo we use [TREC deep learning 2019](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html) queries to test our model. In `data` folder, we provided DL2019 queries and qrel file and also BM25 top1000 results produced by [Anserini](https://github.com/castorini/anserini) (with defualt setting). 
4. To rerank BM25 results, our script can be ran directly by `python3 DeepQLM_T5.py`.

### Results on DL2019
`trec_eval -m ndcg_cut.10 -m map 2019qrels-pass.txt DL2019_bm25_DeepQLM_t5_rerank.res`
```
map                     all     0.4767
ndcg_cut_10             all     0.6569
```

If you want to get MS MARCO dev queries' results, download and unzip qurey file from https://github.com/microsoft/MSMARCO-Passage-Ranking and following [anserini demo](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md) to get BM25 results file. Then change command to `python3 DeepQLM_T5.py --query_path data/queries.dev.small.tsv --run_path data/yourAnseriniBM25run --run_type msmarco`

### To cite our paper
```
@inproceedings{zhuang2021deep,
  author = {Shengyao Zhuang and Hang Li and Guido Zuccon},
  booktitle = {The 43rd European Conference On Information Retrieval (ECIR)},
  title = {Deep Query Likelihood Model for Information Retrieval},
  year = {2021}}
```
