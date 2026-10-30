from retrieval_playbook.config import Settings
from retrieval_playbook.embedders.base import Embedder, EmbedderUnavailable


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedding_base_url:
        from retrieval_playbook.embedders.openai_compat import OpenAICompatEmbedder

        return OpenAICompatEmbedder(
            settings.embedding_base_url,
            settings.embedding_api_key,
            settings.embedding_model or settings.dense_model,
            settings.dense_dim,
        )
    from retrieval_playbook.embedders.jina_local import JinaLocalEmbedder

    return JinaLocalEmbedder(settings.dense_model, settings.dense_dim)


__all__ = ["Embedder", "EmbedderUnavailable", "make_embedder"]
