import xml.etree.ElementTree as ET
from pathlib import Path

from pc import compile_panel


def _panel(path: Path, group_id: str) -> None:
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        f'<rect id="{group_id}" x="0" y="0" width="80" height="60"/>'
        "</svg>"
    )


def _figure(path: Path) -> None:
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 60">'
        '<rect x="0" y="0" width="80" height="60" fill="red"/>'
        "</svg>"
    )


def test_list_config_compiles_all_panels(tmp_path: Path) -> None:
    _panel(tmp_path / "p1.svg", "fig1")
    _panel(tmp_path / "p2.svg", "fig2")
    _figure(tmp_path / "f.svg")
    config = tmp_path / "pc.yaml"
    config.write_text(
        "- panel: p1.svg\n  output: out1.svg\n  fig1: f.svg\n"
        "- panel: p2.svg\n  output: out2.svg\n  fig2: f.svg\n"
    )
    compile_panel(config, tmp_path / "fallback.svg")
    assert (tmp_path / "out1.svg").exists()
    assert (tmp_path / "out2.svg").exists()


def test_list_config_each_panel_gets_own_figure(tmp_path: Path) -> None:
    _panel(tmp_path / "p1.svg", "fig1")
    _panel(tmp_path / "p2.svg", "fig2")
    _figure(tmp_path / "f.svg")
    config = tmp_path / "pc.yaml"
    config.write_text(
        "- panel: p1.svg\n  output: out1.svg\n  fig1: f.svg\n"
        "- panel: p2.svg\n  output: out2.svg\n  fig2: f.svg\n"
    )
    compile_panel(config, tmp_path / "fallback.svg")

    g1 = ET.parse(tmp_path / "out1.svg").getroot().find(".//*[@id='fig1']")
    g2 = ET.parse(tmp_path / "out2.svg").getroot().find(".//*[@id='fig2']")
    assert g1 is not None and list(g1), "panel 1 must have embedded content in fig1"
    assert g2 is not None and list(g2), "panel 2 must have embedded content in fig2"


def test_list_panel_missing_output_key_skips(tmp_path: Path, caplog) -> None:
    import logging
    _panel(tmp_path / "p.svg", "fig")
    _figure(tmp_path / "f.svg")
    config = tmp_path / "pc.yaml"
    # First block has no output key
    config.write_text(
        "- panel: p.svg\n  fig: f.svg\n"
        "- panel: p.svg\n  output: out.svg\n  fig: f.svg\n"
    )
    with caplog.at_level(logging.ERROR, logger="pc"):
        compile_panel(config, tmp_path / "fallback.svg")
    assert (tmp_path / "out.svg").exists(), "valid second block must still compile"
    assert any("output" in r.message for r in caplog.records)
