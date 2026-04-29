import logging
from pathlib import Path

import pytest

from pc import compile_panel


def _make_panel(path: Path) -> None:
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<rect id="fig" x="0" y="0" width="80" height="60"/>'
        "</svg>"
    )


def _make_figure(path: Path) -> None:
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 60">'
        '<rect x="0" y="0" width="80" height="60" fill="blue"/>'
        "</svg>"
    )


def test_duplicate_key_emits_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _make_panel(tmp_path / "panel.svg")
    _make_figure(tmp_path / "fig.svg")
    config = tmp_path / "pc.yaml"
    config.write_text(
        "panel: panel.svg\n"
        "output: out.svg\n"
        "panel: panel.svg\n"  # duplicate
        "fig: fig.svg\n"
    )
    with caplog.at_level(logging.WARNING, logger="pc"):
        compile_panel(config, tmp_path / "fallback.svg")
    assert any("Duplicate YAML key" in r.message for r in caplog.records)


def test_no_warning_for_clean_yaml(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _make_panel(tmp_path / "panel.svg")
    _make_figure(tmp_path / "fig.svg")
    config = tmp_path / "pc.yaml"
    config.write_text("panel: panel.svg\noutput: out.svg\nfig: fig.svg\n")
    with caplog.at_level(logging.WARNING, logger="pc"):
        compile_panel(config, tmp_path / "fallback.svg")
    assert not any("Duplicate YAML key" in r.message for r in caplog.records)
