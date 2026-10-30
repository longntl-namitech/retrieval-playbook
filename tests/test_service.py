import dataclasses

import httpx
import pytest
from fakes import FailingEmbedder

from retrieval_playbook.config import Settings
from retrieval_playbook.service.app import create_app

AUTH = {"Authorization": "Bearer t"}
BODY = {"query": "solar panels", "collection": "c"}


@pytest.fixture
def client_for():
    def make(deps):
        transport = httpx.ASGITransport(app=create_app(deps))
        return httpx.AsyncClient(transport=transport, base_url="http://test")

    return make


async def test_health(deps, client_for):
    async with client_for(deps) as client:
        assert (await client.get("/health")).json() == {"status": "ok"}


async def test_missing_token_config_503(deps, client_for):
    deps = dataclasses.replace(deps, settings=Settings())
    async with client_for(deps) as client:
        assert (await client.post("/search", json=BODY, headers=AUTH)).status_code == 503


async def test_bad_bearer_401(deps, client_for):
    async with client_for(deps) as client:
        bad = {"Authorization": "Bearer nope"}
        assert (await client.post("/search", json=BODY, headers=bad)).status_code == 401
        assert (await client.post("/search", json=BODY)).status_code == 401


async def test_too_many_specs_422(deps, client_for):
    body = {**BODY, "specs": [{"name": f"s{i}"} for i in range(5)]}
    async with client_for(deps) as client:
        assert (await client.post("/search", json=body, headers=AUTH)).status_code == 422


async def test_unknown_method_422(deps, client_for):
    async with client_for(deps) as client:
        response = await client.post("/search", json={**BODY, "method": "nope"}, headers=AUTH)
    assert response.status_code == 422
    assert "hybrid_rrf" in response.text


async def test_embedder_down_503(deps, client_for):
    deps = dataclasses.replace(deps, embedder=FailingEmbedder())
    async with client_for(deps) as client:
        assert (await client.post("/search", json=BODY, headers=AUTH)).status_code == 503


async def test_zero_deadline_504(deps, client_for):
    headers = {**AUTH, "X-Deadline-Ms": "0"}
    async with client_for(deps) as client:
        assert (await client.post("/search", json=BODY, headers=headers)).status_code == 504


async def test_search_ok_shape(deps, client_for):
    async with client_for(deps) as client:
        response = await client.post("/search", json=BODY, headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["results"]["default"]["status"] == "ok"
    assert body["results"]["default"]["rows"][0]["doc_id"] == "d1"


async def test_startup_warms_models(deps, monkeypatch):
    from retrieval_playbook.service import app as app_module

    monkeypatch.setattr(app_module, "build_deps", lambda settings: deps)
    app = create_app()
    async with app.router.lifespan_context(app):
        assert deps.embedder.calls == 1
        assert deps.bm25.calls == 1
        assert app.state.deps is deps
