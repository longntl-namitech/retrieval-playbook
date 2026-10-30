from fakes import FakeEmbedder, FakeSparseEncoder
from qdrant_client import AsyncQdrantClient

from eval.datasets import Dataset, subsample
from eval.index import index_dataset
from retrieval_playbook.config import Settings
from retrieval_playbook.pipeline.search import Deps
from retrieval_playbook.store import Store

DS = Dataset(
    name="toy",
    corpus={f"d{i}": (f"title {i}" if i % 2 else None, f"text number {i}") for i in range(5)},
    queries={"q1": "text number 1"},
    qrels={"q1": {"d1": 1}},
)


def make_deps() -> Deps:
    return Deps(
        settings=Settings(),
        store=Store(AsyncQdrantClient(":memory:")),
        embedder=FakeEmbedder(),
        bm25=FakeSparseEncoder(),
        bm25_vi=FakeSparseEncoder(),
    )


async def test_index_dataset_counts():
    deps = make_deps()
    assert await index_dataset(deps, "toy", DS, batch_size=2) == 5
    assert deps.embedder.calls == 3


async def test_reindex_skips_embedding():
    deps = make_deps()
    await index_dataset(deps, "toy", DS, batch_size=2)
    calls = deps.embedder.calls
    assert await index_dataset(deps, "toy", DS, batch_size=2) == 5
    assert deps.embedder.calls == calls


def test_subsample_is_deterministic():
    ids = [f"q{i}" for i in range(50)]
    assert subsample(ids, 10, 13) == subsample(list(reversed(ids)), 10, 13)
    assert len(subsample(ids, 10, 13)) == 10
    assert subsample(ids, 100, 13) == sorted(ids)
