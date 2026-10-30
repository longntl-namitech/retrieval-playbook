"""FastAPI service: POST /search and GET /health.

Status codes: 401 bad token, 422 bad request or unknown method, 499 client
gone, 503 token not configured or embedder down, 504 deadline spent before
any spec ran. A failing spec is not an HTTP error: it comes back as a row
with status "failed".
"""

from contextlib import asynccontextmanager
from secrets import compare_digest
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Request, Response

from retrieval_playbook.config import Settings
from retrieval_playbook.embedders import EmbedderUnavailable, make_embedder
from retrieval_playbook.models import SearchRequest, SearchResponse
from retrieval_playbook.pipeline.deadline import Deadline
from retrieval_playbook.pipeline.search import DeadlineExpired, Deps, run_search
from retrieval_playbook.registry import get_method


def build_deps(settings: Settings) -> Deps:
    from qdrant_client import AsyncQdrantClient

    from retrieval_playbook.sparse.bm25 import Bm25Encoder, load_vietnamese_stopwords
    from retrieval_playbook.store import Store

    client = AsyncQdrantClient(
        url=settings.qdrant_url, grpc_port=settings.qdrant_grpc_port, prefer_grpc=True
    )
    return Deps(
        settings=settings,
        store=Store(client),
        embedder=make_embedder(settings),
        bm25=Bm25Encoder(),
        bm25_vi=Bm25Encoder(stopwords=load_vietnamese_stopwords(), disable_stemmer=True),
    )


def create_app(deps: Deps | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if deps is None:
            built = build_deps(Settings.from_env())
            # Load models now: a missing model fails at boot, and the first
            # request does not spend its deadline on loading.
            built.bm25.encode_queries(["warm up"])
            await built.embedder.embed_queries(["warm up"])
            app.state.deps = built
        yield

    app = FastAPI(title="retrieval-playbook", lifespan=lifespan)
    app.state.deps = deps

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/search", response_model=SearchResponse)
    async def search(
        req: SearchRequest,
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
        x_deadline_ms: Annotated[int | None, Header()] = None,
    ):
        current: Deps = request.app.state.deps
        settings = current.settings
        if not settings.service_token:
            raise HTTPException(503, "service token not configured")
        if not authorization or not compare_digest(
            authorization, f"Bearer {settings.service_token}"
        ):
            raise HTTPException(401, "invalid token")
        if len(req.specs) > settings.max_specs:
            raise HTTPException(422, f"at most {settings.max_specs} specs")
        try:
            get_method(req.method or settings.default_method)
        except KeyError as exc:
            raise HTTPException(422, exc.args[0]) from None
        budget = settings.default_deadline_ms if x_deadline_ms is None else x_deadline_ms
        if budget <= 0:
            raise HTTPException(504, "deadline already expired")
        deadline = Deadline(min(budget, settings.max_deadline_ms))
        try:
            response = await run_search(req, current, deadline)
        except EmbedderUnavailable:
            raise HTTPException(503, "embedding backend unavailable") from None
        except DeadlineExpired:
            raise HTTPException(504, "deadline expired during embedding") from None
        if await request.is_disconnected():
            return Response(status_code=499)
        return response

    return app


app = create_app()
