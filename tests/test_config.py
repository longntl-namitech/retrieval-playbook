from retrieval_playbook.config import Settings


def test_settings_defaults_and_env_override(monkeypatch):
    monkeypatch.setenv("MAX_SPECS", "2")
    monkeypatch.setenv("SERVICE_TOKEN", "t")

    settings = Settings.from_env()

    assert settings.max_specs == 2
    assert settings.service_token == "t"
    assert settings.dense_dim == 768
    assert settings.default_method == "hybrid_rrf"
