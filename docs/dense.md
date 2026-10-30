# Dense retrieval

## When to use
- Queries are phrased differently from the documents (paraphrases, questions against statements).
- You need one method that is good across domains and languages.

## When not to
- Exact identifiers, codes, article numbers or rare names matter most. Dense models blur them; BM25 keeps them.
- You cannot afford an embedding call per query (it is the slowest leg here, ~20 ms on CPU per query with a 212M model).

## How it works
- [`embedders/jina_local.py`](../src/retrieval_playbook/embedders/jina_local.py): `jina-embeddings-v5-text-nano-retrieval` with separate `query` and `document` prompts, 768 dimensions, normalised, cosine in Qdrant. Loaded in float32 because bfloat16 weights run slowly on most CPUs.
- [`embedders/openai_compat.py`](../src/retrieval_playbook/embedders/openai_compat.py): any OpenAI-compatible `/embeddings` endpoint instead, set `EMBEDDING_BASE_URL`.
- The query is embedded once per request, whatever the number of specs ([`pipeline/search.py`](../src/retrieval_playbook/pipeline/search.py)).

## Results
SciFact:

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `dense` | 0.876 | 0.970 | 0.730 | 0.762 | 21 | 31 |  |  |

TVPL (Vietnamese legal):

| Method | recall@10 | recall@100 | mrr@10 | ndcg@10 | p50 ms | p95 ms | Rescue | Damage |
|---|---|---|---|---|---|---|---|---|
| `dense` | 0.921 | 0.985 | 0.797 | 0.818 | 22 | 30 |  |  |

On TVPL, dense alone is the best method in this repo (see [hybrid fusion](hybrid_fusion.md)).

## Lessons from building this repo
- Load the model before the first request. The first request after starting the service otherwise spent its whole 4 s deadline loading weights and returned 504.
