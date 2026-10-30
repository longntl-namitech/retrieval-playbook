# retrieval-playbook

Hybrid retrieval methods I have used in production, each one runnable as a service, measured on public datasets and written up with what I learned. Work in progress: the service and pipeline are in place; reranking, the eval harness and the method pages are coming next.

## Run it

```bash
cp .env.example .env   # set SERVICE_TOKEN
docker compose up --build
curl -s localhost:8000/health
```

`POST /search` takes a query (or a list of subqueries), a collection, a method name and one or more specs. Each spec is searched independently under the request deadline (`X-Deadline-Ms`); a spec that fails comes back with `status: "failed"` instead of failing the whole request.

## Methods

| Name | What it does |
|---|---|
| `dense` | Dense vectors only |
| `bm25_no_idf` | BM25 without IDF |
| `bm25` | BM25 with IDF |
| `bm25_vi_stopwords` | BM25 with Vietnamese stopwords removed first |
| `hybrid_rrf` | Dense + BM25, reciprocal rank fusion in Qdrant |
| `hybrid_dbsf` | Dense + BM25, distribution-based score fusion |
| `hybrid_weighted_rrf` | Dense + BM25, RRF with the dense leg weighted higher |
| `hybrid_rrf_rerank` | Hybrid RRF, then a cross-encoder rerank |

## Models and licenses

- Dense: [`jinaai/jina-embeddings-v5-text-nano-retrieval`](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval), CC-BY-NC-4.0 (non-commercial). Swap it through the `Embedder` interface or point `EMBEDDING_BASE_URL` at any OpenAI-compatible `/embeddings` endpoint.
- Sparse: fastembed `Qdrant/bm25`.
- Vietnamese stopwords: [stopwords/vietnamese-stopwords](https://github.com/stopwords/vietnamese-stopwords), MIT.

Code: MIT.
