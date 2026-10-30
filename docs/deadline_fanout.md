# Deadline budget and spec fan-out

## When to use
- One request needs several searches with different filters or `k` (for example FAQ entries and documents), and a caller with a hard latency budget.

## How it works
- `X-Deadline-Ms` (default 4,000, max 30,000) creates one budget for the whole request ([`pipeline/deadline.py`](../src/retrieval_playbook/pipeline/deadline.py)).
- The query is embedded once, then each spec runs as its own task under the remaining budget ([`pipeline/search.py`](../src/retrieval_playbook/pipeline/search.py)).
- A spec that raises or runs out of time comes back as `{"status": "failed", "error": "<exception type>"}`; the other specs still return and the HTTP status stays 200. Only the exception type is returned, never its message.
- Status codes for the request itself: 401 bad token, 422 bad request or unknown method, 499 client gone, 503 embedder down or token not configured, 504 budget spent before any spec ran.

## Lessons from building this repo
- Load every model at startup. The first real request returned 504 until the dense model and reranker were warmed up in the service lifespan.
