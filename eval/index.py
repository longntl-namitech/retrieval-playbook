import asyncio

from eval.datasets import Dataset
from retrieval_playbook.pipeline.search import Deps
from retrieval_playbook.store import ChunkPoint


def _document(title: str | None, text: str) -> str:
    return f"{title}\n{text}" if title else text


async def index_dataset(deps: Deps, collection: str, ds: Dataset, batch_size: int = 32) -> int:
    """Index each document as one chunk. A complete collection is left as is."""
    await deps.store.ensure_collection(collection, deps.embedder.dim)
    if await deps.store.count(collection) == len(ds.corpus):
        return len(ds.corpus)
    items = list(ds.corpus.items())
    for n, start in enumerate(range(0, len(items), batch_size), start=1):
        batch = items[start : start + batch_size]
        texts = [_document(title, text) for _, (title, text) in batch]
        dense = await deps.embedder.embed_documents(texts)
        sparse = await asyncio.to_thread(deps.bm25.encode_documents, texts)
        sparse_vi = await asyncio.to_thread(deps.bm25_vi.encode_documents, texts)
        await deps.store.upsert(
            collection,
            [
                ChunkPoint(doc_id, 0, text, dense[i], sparse[i], sparse_vi[i], title=title)
                for i, (doc_id, (title, text)) in enumerate(batch)
            ],
        )
        if n % 20 == 0:
            print(f"indexed {start + len(batch)}/{len(items)}", flush=True)
    return await deps.store.count(collection)
