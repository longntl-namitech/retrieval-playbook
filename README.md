# retrieval-playbook

Retrieval methods I have used in production, rebuilt as one small service, measured on public data and written up with what I learned. Each method is a named entry in a registry: the FastAPI service serves it and the eval harness scores it the same way, so adding a method means adding one entry and running the eval.

## Results

nDCG@10 and the other metrics are averaged over the test queries; latency is in-process on CPU (Apple silicon, one query at a time). Full tables: [SciFact](docs/results/scifact.md), [TVPL](docs/results/tvpl.md).

**SciFact** (English scientific claims, 300 queries, 5,183 documents)

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `dense` | 0.876 | 0.970 | 0.730 | 0.762 | 21 | 31 |  |  |
| `bm25_no_idf` | 0.714 | 0.859 | 0.572 | 0.601 | 7 | 8 |  |  |
| `bm25` | 0.809 | 0.921 | 0.647 | 0.683 | 7 | 16 |  |  |
| `bm25_vi_stopwords` | 0.793 | 0.886 | 0.627 | 0.661 | 6 | 8 |  |  |
| `hybrid_rrf` | 0.882 | 0.973 | 0.718 | 0.754 | 23 | 31 |  |  |
| `hybrid_dbsf` | 0.880 | 0.957 | 0.716 | 0.752 | 22 | 27 |  |  |
| `hybrid_weighted_rrf` | 0.888 | 0.977 | 0.738 | 0.770 | 23 | 31 |  |  |
| `hybrid_rrf_rerank` | 0.736 | 0.973 | 0.615 | 0.642 | 986 | 1257 | 5.0% | 13.0% |

**TVPL** (Vietnamese legal questions, 1,000 sampled queries, 10,576 documents)

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `dense` | 0.921 | 0.985 | 0.797 | 0.818 | 22 | 30 |  |  |
| `bm25_no_idf` | 0.751 | 0.873 | 0.613 | 0.636 | 8 | 9 |  |  |
| `bm25` | 0.800 | 0.915 | 0.641 | 0.669 | 7 | 9 |  |  |
| `bm25_vi_stopwords` | 0.768 | 0.910 | 0.617 | 0.645 | 7 | 11 |  |  |
| `hybrid_rrf` | 0.912 | 0.974 | 0.757 | 0.786 | 26 | 36 |  |  |
| `hybrid_dbsf` | 0.899 | 0.963 | 0.773 | 0.793 | 25 | 32 |  |  |
| `hybrid_weighted_rrf` | 0.919 | 0.978 | 0.784 | 0.808 | 26 | 31 |  |  |
| `hybrid_rrf_rerank` | 0.906 | 0.974 | 0.792 | 0.810 | 812 | 1013 | 11.0% | 4.9% |

What stands out:
- IDF matters: BM25 without it loses 0.03 to 0.08 nDCG@10 ([BM25](docs/bm25.md)).
- A general Vietnamese stopword list made BM25 worse on legal questions, because it removes words that carry meaning there ([BM25](docs/bm25.md)).
- Weighted RRF (dense 1.0, sparse 0.5) is the best hybrid on both sets; on TVPL dense alone is still slightly better ([hybrid fusion](docs/hybrid_fusion.md)).
- The multilingual cross-encoder helps on Vietnamese legal questions and hurts on English scientific claims, at about 1 s per query on CPU ([rerank](docs/rerank.md)).

## Methods

| Name | What it does | Page |
|---|---|---|
| `dense` | Jina v5 nano embeddings | [dense](docs/dense.md) |
| `bm25_no_idf` | BM25 without IDF | [BM25](docs/bm25.md) |
| `bm25` | BM25 with IDF | [BM25](docs/bm25.md) |
| `bm25_vi_stopwords` | BM25, Vietnamese stopwords removed first | [BM25](docs/bm25.md) |
| `hybrid_rrf` | Dense + BM25, reciprocal rank fusion in Qdrant | [hybrid fusion](docs/hybrid_fusion.md) |
| `hybrid_dbsf` | Dense + BM25, distribution-based score fusion | [hybrid fusion](docs/hybrid_fusion.md) |
| `hybrid_weighted_rrf` | Dense + BM25, RRF with dense weighted 2:1 | [hybrid fusion](docs/hybrid_fusion.md) |
| `hybrid_rrf_rerank` | Hybrid RRF, then a cross-encoder over the top candidates | [rerank](docs/rerank.md) |

Service features shared by all methods: [deadline budget and spec fan-out](docs/deadline_fanout.md), [multi-query merge](docs/multi_query.md), [context expansion](docs/context_expansion.md).

## Quickstart

```bash
cp .env.example .env            # set SERVICE_TOKEN
docker compose up -d --build    # Qdrant + the service on :8000

# index SciFact and score every method (downloads the dataset; ~5 min to index on CPU)
uv sync --extra eval
uv run --env-file .env python -m eval.run --dataset scifact --query-limit 20 --name smoke

curl -s localhost:8000/search \
  -H "Authorization: Bearer $SERVICE_TOKEN" -H "X-Deadline-Ms: 2000" \
  -H "content-type: application/json" \
  -d '{"query": "vitamin D and bone density", "collection": "eval-scifact",
       "method": "hybrid_weighted_rrf", "specs": [{"name": "top", "k": 5}]}'
```

The service needs `SERVICE_TOKEN` set. Each spec in a request runs independently; a spec that fails comes back with `"status": "failed"` and the rest still return.

## Adding a method

1. If it needs a new kind of vector or fusion, add it to [`fusion.py`](src/retrieval_playbook/fusion.py) and [`store.py`](src/retrieval_playbook/store.py); a new reranker implements the `Reranker` protocol in [`rerankers/`](src/retrieval_playbook/rerankers/).
2. Add a `MethodConfig` to [`registry.py`](src/retrieval_playbook/registry.py).
3. Run `python -m eval.run` on both datasets, then `python -m eval.report`, and add a page under `docs/`.

## Tests

```bash
uv run pytest                  # no network, no Qdrant server
uv run pytest -m integration   # loads the real models
```

## Roadmap

Query rewriting and speculative search, LLM reranking (with a token estimate before each run), result caching, Qdrant load tests and quantization, semantic chunking, embedding model comparison, GPU reranker serving.

## Models, data and licenses

- Dense: [`jinaai/jina-embeddings-v5-text-nano-retrieval`](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval), CC-BY-NC-4.0 (non-commercial). Swap it through the `Embedder` protocol or point `EMBEDDING_BASE_URL` at an OpenAI-compatible `/embeddings` endpoint.
- Sparse: fastembed `Qdrant/bm25`. Rerank: [`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1).
- Vietnamese stopwords: [stopwords/vietnamese-stopwords](https://github.com/stopwords/vietnamese-stopwords), MIT.
- Datasets are downloaded at runtime and not stored here: [BeIR/scifact](https://huggingface.co/datasets/BeIR/scifact) and [GreenNode/TVPL-Retrieval-VN](https://huggingface.co/datasets/GreenNode/TVPL-Retrieval-VN), both CC-BY-SA-4.0.

Code: MIT.
