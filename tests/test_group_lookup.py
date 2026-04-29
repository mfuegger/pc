from pathlib import Path

from pc import _compile_tree

NS_INK = "http://www.inkscape.org/namespaces/inkscape"
NS_SVG = "http://www.w3.org/2000/svg"


def _figure(path: Path) -> None:
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 50 50">'
        '<rect x="0" y="0" width="50" height="50" fill="green"/>'
        "</svg>"
    )


def _panel_with(path: Path, extra_attrs: str) -> None:
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<svg xmlns="{NS_SVG}" xmlns:inkscape="{NS_INK}" viewBox="0 0 100 100">'
        f'<g {extra_attrs} width="60" height="60"/>'
        "</svg>"
    )


def test_lookup_by_inkscape_label(tmp_path: Path) -> None:
    _figure(tmp_path / "f.svg")
    _panel_with(tmp_path / "p.svg", f'inkscape:label="myplot"')
    tree = _compile_tree({"panel": "p.svg", "myplot": "f.svg"}, tmp_path / "pc.yaml")
    assert tree is not None
    g = tree.getroot().find(f".//*[@{{{NS_INK}}}label='myplot']")
    assert g is not None and list(g)


def test_lookup_by_label_attribute(tmp_path: Path) -> None:
    _figure(tmp_path / "f.svg")
    _panel_with(tmp_path / "p.svg", 'label="myplot"')
    tree = _compile_tree({"panel": "p.svg", "myplot": "f.svg"}, tmp_path / "pc.yaml")
    assert tree is not None
    g = tree.getroot().find(".//*[@label='myplot']")
    assert g is not None and list(g)


def test_lookup_by_id(tmp_path: Path) -> None:
    _figure(tmp_path / "f.svg")
    _panel_with(tmp_path / "p.svg", 'id="myplot"')
    tree = _compile_tree({"panel": "p.svg", "myplot": "f.svg"}, tmp_path / "pc.yaml")
    assert tree is not None
    g = tree.getroot().find(".//*[@id='myplot']")
    assert g is not None and list(g)
