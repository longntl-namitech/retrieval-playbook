from retrieval_playbook.config import Settings
from retrieval_playbook.rerankers.base import Reranker


def make_reranker(settings: Settings) -> Reranker:
    if settings.rerank_base_url and settings.rerank_api_key:
        from retrieval_playbook.rerankers.cohere_http import CohereHttpReranker

        return CohereHttpReranker(
            settings.rerank_base_url, settings.rerank_api_key, settings.rerank_model or ""
        )
    from retrieval_playbook.rerankers.cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker(settings.cross_encoder_model)


__all__ = ["Reranker", "make_reranker"]
