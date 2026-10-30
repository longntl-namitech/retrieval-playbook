"""The response shape is a contract for callers (e.g. the RAG reference stack)."""

import json
from pathlib import Path

import httpx

from retrieval_playbook.service.app import create_app

GOLDEN = json.loads((Path(__file__).parent / "contract" / "search_basic.json").read_text())


async def test_search_response_matches_contract(deps):
    transport = httpx.ASGITransport(app=create_app(deps))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/search", json=GOLDEN["request"], headers={"Authorization": "Bearer t"}
        )
    body = response.json()
    assert response.status_code == 200
    assert list(body) == GOLDEN["response_keys"]
    assert list(body["timing"]) == GOLDEN["timing_keys"]
    assert list(body["results"]) == ["energy", "all"]
    for spec in body["results"].values():
        assert list(spec) == GOLDEN["spec_keys"]
        assert spec["status"] == "ok"
        for row in spec["rows"]:
            assert list(row) == GOLDEN["row_keys"]
