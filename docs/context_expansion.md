# Context expansion

## When to use
- Chunks are small and the answer spans the chunk boundary, or a short FAQ entry should come back whole.

## How it works
- A spec can ask for `context: {"prev": n, "next": m}` or `{"full": true}` ([`pipeline/context.py`](../src/retrieval_playbook/pipeline/context.py)).
- Chunks are stored with `doc_id` and `chunk_index`; neighbours are fetched with one filtered scroll per hit, after ranking, so they never affect scores.

## Results
No public numbers. The eval datasets index each document as one passage, so there are no neighbours to add. Behaviour is covered by tests (`test_expand_prev_next`, `test_expand_full_document`, `test_context_expansion_applied`).
