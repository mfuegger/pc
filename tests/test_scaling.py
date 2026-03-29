"""Tests for figure scaling into panel placeholders.

Covers the full range of SVG unit conventions that appear in practice:
  - pt-unit sources (pdf2svg output)
  - mm-unit sources (Inkscape exports)
  - unitless sources (hand-crafted SVGs)
  - unitless rect targets in a mm-panel template

The invariant: after embedding, a figure must fill its placeholder rect to
within rounding tolerance, regardless of the physical units on the source SVG.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from pc import SVGDimensions, _compile_tree, compile_panel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _figure_svg(
    path: Path, vb_w: float, vb_h: float, width_attr: str, height_attr: str
) -> None:
    """Write a minimal source SVG with the given viewBox and width/height attrs."""
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg"'
        f' width="{width_attr}" height="{height_attr}"'
        f' viewBox="0 0 {vb_w} {vb_h}">'
        f'<rect id="r" x="0" y="0" width="{vb_w}" height="{vb_h}" fill="blue"/>'
        "</svg>"
    )


def _panel_mm(
    path: Path, rect_x: float, rect_y: float, rect_w: float, rect_h: float
) -> None:
    """Write a panel template where 1 user unit = 1 mm, with one rect placeholder."""
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg"'
        ' width="175mm" height="68mm" viewBox="0 0 175 68">'
        f'<rect id="fig" x="{rect_x}" y="{rect_y}" width="{rect_w}" height="{rect_h}"'
        ' fill="#f0f0f0"/>'
        "</svg>"
    )


def _extract_scale(tree: "ET.ElementTree[ET.Element]") -> float:
    """Return the scale factor applied to the first child of the 'fig' group."""
    g = tree.getroot().find(".//*[@id='fig']")
    assert g is not None, "group 'fig' not found"
    children = list(g)
    assert children, "group 'fig' has no children"
    transform = children[0].get("transform", "")
    assert "scale(" in transform, f"no scale transform found: {transform!r}"
    return float(transform.split("scale(")[1].split(")")[0])


# ---------------------------------------------------------------------------
# SVGDimensions.from_svg — unit handling
# ---------------------------------------------------------------------------


def test_from_svg_pt_units_returns_viewbox_user_units(tmp_path: Path) -> None:
    """pdf2svg output: 595pt x 420pt with viewBox 0 0 595 420 -> user units (595, 420)."""
    _figure_svg(tmp_path / "fig.svg", 595, 420, "595pt", "420pt")
    dims = SVGDimensions.from_svg(tmp_path / "fig.svg")
    assert dims.width == pytest.approx(595)
    assert dims.height == pytest.approx(420)


def test_from_svg_mm_units_returns_viewbox_user_units(tmp_path: Path) -> None:
    """Inkscape-style SVG: 175mm x 68mm with viewBox 0 0 175 68 -> user units (175, 68)."""
    _figure_svg(tmp_path / "fig.svg", 175, 68, "175mm", "68mm")
    dims = SVGDimensions.from_svg(tmp_path / "fig.svg")
    assert dims.width == pytest.approx(175)
    assert dims.height == pytest.approx(68)


def test_from_svg_in_units_returns_viewbox_user_units(tmp_path: Path) -> None:
    _figure_svg(tmp_path / "fig.svg", 460.8, 345.6, "6.4in", "4.8in")
    dims = SVGDimensions.from_svg(tmp_path / "fig.svg")
    assert dims.width == pytest.approx(460.8)
    assert dims.height == pytest.approx(345.6)


def test_from_svg_no_viewbox_unitless(tmp_path: Path) -> None:
    """No viewBox, unitless width/height: return the raw numbers as user units."""
    (tmp_path / "fig.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"/>'
    )
    dims = SVGDimensions.from_svg(tmp_path / "fig.svg")
    assert dims.width == pytest.approx(200)
    assert dims.height == pytest.approx(100)


def test_from_svg_no_viewbox_physical_units_raises(tmp_path: Path) -> None:
    """Without a viewBox, element coordinates are ambiguous given physical units."""
    (tmp_path / "fig.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="200mm" height="100mm"/>'
    )
    with pytest.raises(ValueError, match="ambiguous"):
        SVGDimensions.from_svg(tmp_path / "fig.svg")


# ---------------------------------------------------------------------------
# End-to-end scale correctness
# ---------------------------------------------------------------------------


def test_scale_pt_source_into_mm_panel(tmp_path: Path) -> None:
    """pt-unit source (pdf2svg) into a mm-panel rect: scale uses viewBox user units.

    Source 595x420 uu, target rect 84x64 uu (mm), contain fit:
      scale = min(84/595, 64/420) = 84/595 ≈ 0.1412
    """
    _figure_svg(tmp_path / "fig.svg", 595, 420, "595pt", "420pt")
    _panel_mm(tmp_path / "panel.svg", 2, 2, 84, 64)

    tree = _compile_tree({"panel": "panel.svg", "fig": "fig.svg"}, tmp_path / "pc.yaml")
    assert tree is not None
    assert _extract_scale(tree) == pytest.approx(84 / 595, rel=1e-3)


def test_scale_mm_source_into_mm_panel(tmp_path: Path) -> None:
    """mm-unit source into mm-panel rect: scale is the viewBox ratio, not an mm/mm ratio.

    Source 200x100 uu (mm), target 100x50 uu (mm): scale = 0.5.
    """
    _figure_svg(tmp_path / "fig.svg", 200, 100, "200mm", "100mm")
    _panel_mm(tmp_path / "panel.svg", 0, 0, 100, 50)

    tree = _compile_tree({"panel": "panel.svg", "fig": "fig.svg"}, tmp_path / "pc.yaml")
    assert tree is not None
    assert _extract_scale(tree) == pytest.approx(0.5, rel=1e-3)


def test_scale_unitless_source_exact(tmp_path: Path) -> None:
    """Unitless source 400x100 into 200x100 rect with fit=width: scale exactly 0.5."""
    _figure_svg(tmp_path / "fig.svg", 400, 100, "400", "100")
    _panel_mm(tmp_path / "panel.svg", 0, 0, 200, 100)

    tree = _compile_tree(
        {"panel": "panel.svg", "fig": {"file": "fig.svg", "fit": "width"}},
        tmp_path / "pc.yaml",
    )
    assert tree is not None
    assert _extract_scale(tree) == pytest.approx(0.5, rel=1e-3)


def test_scale_height_fit(tmp_path: Path) -> None:
    """fit=height: scale = target_h / source_h regardless of width."""
    _figure_svg(tmp_path / "fig.svg", 200, 50, "200", "50")
    _panel_mm(tmp_path / "panel.svg", 0, 0, 80, 100)

    tree = _compile_tree(
        {"panel": "panel.svg", "fig": {"file": "fig.svg", "fit": "height"}},
        tmp_path / "pc.yaml",
    )
    assert tree is not None
    assert _extract_scale(tree) == pytest.approx(100 / 50, rel=1e-3)


# ---------------------------------------------------------------------------
# Rect-placeholder replacement
# ---------------------------------------------------------------------------


def test_rect_replaced_by_translated_group(tmp_path: Path) -> None:
    """Rect placeholder must become a <g transform='translate(x,y)'> with content."""
    _figure_svg(tmp_path / "fig.svg", 100, 100, "100", "100")
    _panel_mm(tmp_path / "panel.svg", 10, 20, 80, 60)

    tree = _compile_tree({"panel": "panel.svg", "fig": "fig.svg"}, tmp_path / "pc.yaml")
    assert tree is not None
    root = tree.getroot()

    # Original rect is gone
    assert root.find(".//*[@id='fig'][@width]") is None

    g = root.find(".//*[@id='fig']")
    assert g is not None
    transform = g.get("transform") or ""
    assert transform == "translate(10,20)"

    assert list(g), "group must contain embedded content"


def test_no_rect_width_attr_after_compile(tmp_path: Path) -> None:
    """The group replacing the rect must not carry over the width/height attrs."""
    _figure_svg(tmp_path / "fig.svg", 100, 80, "100", "80")
    _panel_mm(tmp_path / "panel.svg", 5, 5, 84, 60)

    tree = _compile_tree({"panel": "panel.svg", "fig": "fig.svg"}, tmp_path / "pc.yaml")
    assert tree is not None
    g = tree.getroot().find(".//*[@id='fig']")
    assert g is not None
    assert g.get("width") is None
    assert g.get("height") is None


# ---------------------------------------------------------------------------
# Multi-output: compile once, write many
# ---------------------------------------------------------------------------


def test_multi_output_identical_content(tmp_path: Path) -> None:
    """Two outputs from one config block must be byte-identical (compiled once)."""
    _figure_svg(tmp_path / "fig.svg", 100, 100, "100", "100")
    _panel_mm(tmp_path / "panel.svg", 0, 0, 80, 60)
    config = tmp_path / "pc.yaml"
    config.write_text("panel: panel.svg\noutput: [out1.svg, out2.svg]\nfig: fig.svg\n")

    compile_panel(config, tmp_path / "fallback.svg")

    assert (tmp_path / "out1.svg").read_text() == (tmp_path / "out2.svg").read_text()


def test_template_not_overwritten_when_output_differs(tmp_path: Path) -> None:
    """Template file must be unchanged when output path differs from template path."""
    _figure_svg(tmp_path / "fig.svg", 100, 100, "100", "100")
    _panel_mm(tmp_path / "panel_tpl.svg", 0, 0, 80, 60)
    original = (tmp_path / "panel_tpl.svg").read_text()
    config = tmp_path / "pc.yaml"
    config.write_text("panel: panel_tpl.svg\noutput: out.svg\nfig: fig.svg\n")

    compile_panel(config, tmp_path / "fallback.svg")

    assert (tmp_path / "panel_tpl.svg").read_text() == original


def test_panel_is_template_and_output_does_not_corrupt(tmp_path: Path) -> None:
    """When panel == output, the second call must use the original template tree."""
    _figure_svg(tmp_path / "fig.svg", 100, 100, "100", "100")
    _panel_mm(tmp_path / "panel.svg", 0, 0, 80, 60)
    config = tmp_path / "pc.yaml"
    config.write_text("panel: panel.svg\noutput: [panel.svg, out2.svg]\nfig: fig.svg\n")

    compile_panel(config, tmp_path / "fallback.svg")

    # Both outputs must have embedded content (not the raw template)
    for name in ("panel.svg", "out2.svg"):
        root = ET.parse(tmp_path / name).getroot()
        g = root.find(".//*[@id='fig']")
        assert g is not None, f"{name}: group 'fig' missing"
        assert list(g), f"{name}: group 'fig' has no content"

    # And they must agree
    assert (tmp_path / "panel.svg").read_text() == (tmp_path / "out2.svg").read_text()
