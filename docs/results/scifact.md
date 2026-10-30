# scifact

300 queries, 5183 documents, run 2026-10-30. Dense `jinaai/jina-embeddings-v5-text-nano-retrieval`, sparse `Qdrant/bm25`, rerank `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` over the top 100. Latency is in-process on CPU, one query at a time.

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
