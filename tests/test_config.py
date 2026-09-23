from pathlib import Path

import pytest

from config import RuntimeConfig, load_config


def write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_repository_runtime_config() -> None:
    config = load_config("config.yaml")

    assert isinstance(config, RuntimeConfig)
    assert config.model.name_or_path
    assert config.engine.max_new_tokens <= config.engine.max_model_len
    assert config.sampling.top_k >= 0


def test_uses_defaults_for_optional_sections(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        """
model:
  name_or_path: local-model
""",
    )

    config = load_config(path)

    assert config.engine.max_new_tokens == 128
    assert config.sampling.top_k == 0
    assert config.sampling.top_p == 0.9


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            """
model:
  name_or_path: local-model
  unknown_field: true
""",
            "unknown keys in model",
        ),
        (
            """
model:
  name_or_path: local-model
  dtype: int8
""",
            "model.dtype",
        ),
        (
            """
model:
  name_or_path: local-model
engine:
  max_model_len: 32
  max_new_tokens: 64
""",
            "max_new_tokens",
        ),
        (
            """
model:
  name_or_path: local-model
sampling:
  temperature: -1
""",
            "temperature",
        ),
    ],
)
def test_rejects_invalid_values(tmp_path: Path, content: str, message: str) -> None:
    path = write_config(tmp_path, content)

    with pytest.raises((TypeError, ValueError), match=message):
        load_config(path)


def test_rejects_unknown_top_level_section(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        """
model:
  name_or_path: local-model
scheduler: {}
""",
    )

    with pytest.raises(ValueError, match="unknown keys in config root"):
        load_config(path)
