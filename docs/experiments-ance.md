# Pyserini: Reproducing ANCE Results

This guide provides instructions to reproduce the following dense retrieval work:

> Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, Arnold Overwijk. [Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval](https://arxiv.org/pdf/2007.00808.pdf)

Note that we often observe minor differences in scores between different computing environments (e.g., Linux vs. macOS).
However, the differences usually appear in the fifth digit after the decimal point, and do not appear to be a cause for concern from a reproducibility perspective.
Thus, while the scoring script provides results to much higher precision, we have intentionally rounded to four digits after the decimal point.

## MS MARCO Passage

**ANCE retrieval** with brute-force index:

```bash
python -m pyserini.search.faiss \
  --index msmarco-v1-passage.ance \
  --topics msmarco-passage-dev-subset \
  --encoded-queries ance-msmarco-passage-dev-subset \
  --output runs/run.msmarco-passage.ance.tsv \
  --output-format msmarco \
  --batch-size 512 --threads 16
```

The option `--encoded-queries` specifies the use of encoded queries (i.e., queries that have already been converted into dense vectors and cached).
As an alternative, replace with `--encoder castorini/ance-msmarco-passage` to perform "on-the-fly" query encoding, i.e., convert text queries into dense vectors as part of the dense retrieval process.

To evaluate:

```bash
python -m pyserini.eval.msmarco_passage_eval msmarco-passage-dev-subset \
  runs/run.msmarco-passage.ance.tsv
```

Results:

```
#####################
MRR @10: 0.3302
QueriesRanked: 6980
#####################
```

We can also use the official TREC evaluation tool `trec_eval` to compute other metrics than MRR@10. 
For that we first need to convert runs and qrels files to the TREC format:

```bash
python -m pyserini.eval.convert_msmarco_run_to_trec_run \
  --input runs/run.msmarco-passage.ance.tsv \
  --output runs/run.msmarco-passage.ance.trec

python -m pyserini.eval.trec_eval -c -mrecall.1000 -mmap msmarco-passage-dev-subset \
    runs/run.msmarco-passage.ance.trec
```

Results:

```
map                   	all	0.3363
recall_1000           	all	0.9584
```

## MS MARCO Document

**ANCE retrieval** with brute-force index:

```bash
python -m pyserini.search.faiss \
  --index msmarco-v1-doc.ance-maxp \
  --topics msmarco-doc-dev \
  --encoded-queries ance_maxp-msmarco-doc-dev \
  --output runs/run.msmarco-doc.passage.ance-maxp.txt \
  --output-format msmarco \
  --batch-size 512 --threads 16 \
  --hits 1000 --max-passage --max-passage-hits 100
```

Same as above, replace `--encoded-queries` with `--encoder castorini/ance-msmarco-doc-maxp` for on-the-fly query encoding.

To evaluate:

```bash
python -m pyserini.eval.msmarco_doc_eval \
  --judgments msmarco-doc-dev \
  --run runs/run.msmarco-doc.passage.ance-maxp.txt
```

Results:

```
#####################
MRR @100: 0.3794
QueriesRanked: 5193
#####################
```

We can also use the official TREC evaluation tool `trec_eval` to compute other metrics than MRR@100. 
For that we first need to convert runs and qrels files to the TREC format:

```bash
python -m pyserini.eval.convert_msmarco_run_to_trec_run \
  --input runs/run.msmarco-doc.passage.ance-maxp.txt \
  --output runs/run.msmarco-doc.passage.ance-maxp.trec

python -m pyserini.eval.trec_eval -c -mrecall.100 -mmap msmarco-doc-dev \
  runs/run.msmarco-doc.passage.ance-maxp.trec
```

Results:

```
map                   	all	0.3794
recall_100            	all	0.9033
```

## Natural Questions (NQ)

**ANCE retrieval** with brute-force index:

```bash
python -m pyserini.search.faiss \
  --index wikipedia-dpr-100w.ance-multi \
  --topics dpr-nq-test \
  --encoded-queries ance_multi-nq-test \
  --output runs/run.ance.nq-test.multi.trec \
  --batch-size 512 --threads 16
```

Same as above, replace `--encoded-queries` with `--encoder castorini/ance-dpr-question-multi` for on-the-fly query encoding.

To evaluate, first convert the TREC output format to DPR's `json` format:

```bash
python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
  --topics dpr-nq-test \
  --index wikipedia-dpr \
  --input runs/run.ance.nq-test.multi.trec \
  --output runs/run.ance.nq-test.multi.json

python -m pyserini.eval.evaluate_dpr_retrieval \
  --retrieval runs/run.ance.nq-test.multi.json \
  --topk 20 100
```

Results:

```
Top20	accuracy: 0.8224
Top100	accuracy: 0.8787
```

## Trivia QA

**ANCE retrieval** with brute-force index:

```bash
python -m pyserini.search.faiss \
  --index wikipedia-dpr-100w.ance-multi \
  --topics dpr-trivia-test \
  --encoded-queries ance_multi-trivia-test \
  --output runs/run.ance.trivia-test.multi.trec \
  --batch-size 512 --threads 16
```

Same as above, replace `--encoded-queries` with `--encoder castorini/ance-dpr-question-multi` for on-the-fly query encoding.

To evaluate, first convert the TREC output format to DPR's `json` format:

```bash
python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
  --topics dpr-trivia-test \
  --index wikipedia-dpr \
  --input runs/run.ance.trivia-test.multi.trec \
  --output runs/run.ance.trivia-test.multi.json

python -m pyserini.eval.evaluate_dpr_retrieval \
  --retrieval runs/run.ance.trivia-test.multi.json \
  --topk 20 100
```

Results:

```
Top20	accuracy: 0.8010
Top100	accuracy: 0.8522
```

## Reproduction Log[*](reproducibility.md)

+ Results reproduced by [@lintool](https://github.com/lintool) on 2021-04-25 (commit [`854c19`](https://github.com/castorini/pyserini/commit/854c1930ba00819245c0a9fbcf2090ce14db4db0))
+ Results reproduced by [@jingtaozhan](https://github.com/jingtaozhan) on 2021-05-15 (commit [`53d8d3`](https://github.com/castorini/pyserini/commit/53d8d3cbb78c88a23ce132a42b0396caad7d2e0f))
+ Results reproduced by [@jmmackenzie](https://github.com/jmmackenzie) on 2021-05-17 (PyPI [`0.12.0`](https://pypi.org/project/pyserini/0.12.0/))
+ Results reproduced by [@yuki617](https://github.com/yuki617) on 2021-06-07 (commit [`c7b37d`](https://github.com/castorini/pyserini/commit/c7b37d6073cda62685f64d6d0b99dc46f0718346))
+ Results reproduced by [@ArthurChen189](https://github.com/ArthurChen189) on 2021-07-06 (commit [`c9f44b`](https://github.com/castorini/pyserini/commit/c9f44b2a24103fff4887cade831f9b7c2472b190))
+ Results reproduced by [@lintool](https://github.com/lintool) on 2022-12-23 (commit [`0c495c`](https://github.com/castorini/pyserini/commit/0c495cf2999dda980eb1f85efa30a4323cef5855))
+ Results reproduced by [@lintool](https://github.com/lintool) on 2023-01-10 (commit [`7dafc4`](https://github.com/castorini/pyserini/commit/7dafc4f918bd44ada3771a5c81692ab19cc2cae9))
+ Results reproduced by [@lintool](https://github.com/lintool) on 2024-10-07 (commit [`3f7609`](https://github.com/castorini/pyserini/commit/3f76099a73820afee12496c0354d52ca6a6175c2))
