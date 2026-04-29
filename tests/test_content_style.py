import xml.etree.ElementTree as ET
from pathlib import Path

from pc import _compile_tree


def _panel(path: Path) -> None:
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<rect id="fig" x="0" y="0" width="80" height="60"/>'
        "</svg>"
    )


def _figure(path: Path) -> None:
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 60">'
        '<rect x="0" y="0" width="80" height="60" fill="blue"/>'
        "</svg>"
    )


def _style_text(tree: "ET.ElementTree[ET.Element]") -> str:
    ns = "{http://www.w3.org/2000/svg}"
    root = tree.getroot()
    styles = root.findall(f".//{ns}style") + root.findall(".//style")
    return "\n".join(s.text or "" for s in styles)


def test_default_content_style_injected(tmp_path: Path) -> None:
    _panel(tmp_path / "p.svg")
    _figure(tmp_path / "f.svg")
    tree = _compile_tree({"panel": "p.svg", "fig": "f.svg"}, tmp_path / "pc.yaml")
    assert tree is not None
    assert ".pc-content" in _style_text(tree)


def test_custom_content_style_applied(tmp_path: Path) -> None:
    _panel(tmp_path / "p.svg")
    _figure(tmp_path / "f.svg")
    tree = _compile_tree(
        {"panel": "p.svg", "content_style": "stroke: red;", "fig": "f.svg"},
        tmp_path / "pc.yaml",
    )
    assert tree is not None
    assert "stroke: red;" in _style_text(tree)


def test_tex_content_style_always_stroke_none(tmp_path: Path) -> None:
    """pc-tex-content must always get stroke:none regardless of content_style."""
    _panel(tmp_path / "p.svg")
    _figure(tmp_path / "f.svg")
    tree = _compile_tree(
        {"panel": "p.svg", "content_style": "stroke: red; fill: initial;", "fig": "f.svg"},
        tmp_path / "pc.yaml",
    )
    assert tree is not None
    style = _style_text(tree)
    assert "stroke: none" in style
    assert ".pc-tex-content" in style


def test_style_not_accumulated_on_recompile(tmp_path: Path) -> None:
    """Recompiling a self-overwriting panel must not duplicate the style block."""
    _panel(tmp_path / "p.svg")
    _figure(tmp_path / "f.svg")
    config = {"panel": "p.svg", "output": "p.svg", "fig": "f.svg"}

    from pc import _write_output
    tree = _compile_tree(config, tmp_path / "pc.yaml")
    assert tree is not None
    _write_output(tree, tmp_path / "p.svg")

    # Second compile on the already-compiled SVG
    config["panel"] = "p.svg"
    tree2 = _compile_tree(config, tmp_path / "pc.yaml")
    assert tree2 is not None
    style = _style_text(tree2)
    assert style.count(".pc-content path") == 1, "style block must not accumulate"
