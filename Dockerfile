FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy HF_HOME=/opt/hf FASTEMBED_CACHE_PATH=/opt/fastembed

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Bake the models into the image so the service starts without network access.
RUN uv run --no-sync python -c "\
from retrieval_playbook.config import Settings; \
from retrieval_playbook.embedders.jina_local import JinaLocalEmbedder; \
from retrieval_playbook.sparse.bm25 import Bm25Encoder; \
from sentence_transformers import CrossEncoder; \
s = Settings(); \
JinaLocalEmbedder(s.dense_model)._load(); \
Bm25Encoder().encode_queries(['warm up']); \
CrossEncoder(s.cross_encoder_model)"

ENV HF_HUB_OFFLINE=1
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "retrieval_playbook.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
