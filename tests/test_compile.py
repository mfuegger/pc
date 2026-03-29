import xml.etree.ElementTree as ET
from pathlib import Path

from pc import _compile_one, compile_panel

INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"


def _make_panel(
    path: Path, label: str = "plot", width: int = 200, height: int = 100
) -> None:
    path.write_text(
        f"<?xml version='1.0' encoding='utf-8'?>"
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="{INKSCAPE_NS}"'
        f' width="400" height="300" viewBox="0 0 400 300">'
        f'<g inkscape:label="{label}" width="{width}" height="{height}"/>'
        f"</svg>"
    )


def _make_figure(path: Path, width: int = 200, height: int = 100) -> None:
    path.write_text(
        f"<?xml version='1.0' encoding='utf-8'?>"
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect id="bg" x="0" y="0" width="{width}" height="{height}" fill="blue"/>'
        f"</svg>"
    )


def test_compile_svg_figure(tmp_path: Path) -> None:
    _make_panel(tmp_path / "panel.svg")
    _make_figure(tmp_path / "fig.svg")
    output = tmp_path / "out.svg"

    _compile_one(
        {"panel": "panel.svg", "plot": {"file": "fig.svg", "fit": "contain"}},
        tmp_path / "pc.yaml",
        output,
    )

    assert output.exists()
    group = ET.parse(output).getroot().find(f".//*[@{{{INKSCAPE_NS}}}label='plot']")
    assert group is not None
    assert len(list(group)) > 0


def test_compile_shorthand_string(tmp_path: Path) -> None:
    _make_panel(tmp_path / "panel.svg")
    _make_figure(tmp_path / "fig.svg")
    output = tmp_path / "out.svg"

    _compile_one(
        {"panel": "panel.svg", "plot": "fig.svg"}, tmp_path / "pc.yaml", output
    )

    assert output.exists()


def test_scale_applied(tmp_path: Path) -> None:
    """Figure 400x100 into group 200x100 with fit=width → scale 0.5."""
    _make_panel(tmp_path / "panel.svg", width=200, height=100)
    _make_figure(tmp_path / "fig.svg", width=400, height=100)
    output = tmp_path / "out.svg"

    _compile_one(
        {"panel": "panel.svg", "plot": {"file": "fig.svg", "fit": "width"}},
        tmp_path / "pc.yaml",
        output,
    )

    group = ET.parse(output).getroot().find(f".//*[@{{{INKSCAPE_NS}}}label='plot']")
    assert group is not None
    assert "scale(0.5)" in (list(group)[0].get("transform") or "")


def test_missing_group_skips_gracefully(tmp_path: Path) -> None:
    _make_panel(tmp_path / "panel.svg", label="plot")
    _make_figure(tmp_path / "fig.svg")
    output = tmp_path / "out.svg"

    # "other" label doesn't exist — should not crash, output still written
    _compile_one(
        {"panel": "panel.svg", "other": "fig.svg"}, tmp_path / "pc.yaml", output
    )

    assert output.exists()


def test_multi_output(tmp_path: Path) -> None:
    _make_panel(tmp_path / "panel.svg")
    _make_figure(tmp_path / "fig.svg")
    config = tmp_path / "pc.yaml"
    config.write_text(
        "panel: panel.svg\noutput:\n  - out1.svg\n  - out2.svg\nplot:\n  file: fig.svg\n"
    )

    compile_panel(config, tmp_path / "fallback.svg")

    assert (tmp_path / "out1.svg").exists()
    assert (tmp_path / "out2.svg").exists()
