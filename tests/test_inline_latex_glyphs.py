import xml.etree.ElementTree as ET

from pc import _inline_latex_glyphs


NS = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"


def _glyph_svg() -> list[ET.Element]:
    """Minimal inkscape-style LaTeX SVG: one glyph in defs, one <use> referencing it."""
    root = ET.fromstring(
        f'<svg xmlns="{NS}" xmlns:xlink="{XLINK}">'
        f'  <defs>'
        f'    <g id="glyph-0">'
        f'      <path id="path1" d="M 0 0 L 10 10"/>'
        f'    </g>'
        f'  </defs>'
        f'  <g fill="black">'
        f'    <use xlink:href="#glyph-0" x="5" y="12"/>'
        f'  </g>'
        f'</svg>'
    )
    return list(root)


def test_use_replaced_by_inline_group() -> None:
    elements = _glyph_svg()
    _inline_latex_glyphs(elements)
    tags = {el.tag.split("}")[-1] if "}" in el.tag else el.tag for el in _iter_all(elements)}
    assert "use" not in tags


def test_defs_removed() -> None:
    elements = _glyph_svg()
    _inline_latex_glyphs(elements)
    tags = {el.tag.split("}")[-1] if "}" in el.tag else el.tag for el in _iter_all(elements)}
    assert "defs" not in tags


def test_path_inlined_with_translate() -> None:
    elements = _glyph_svg()
    _inline_latex_glyphs(elements)
    # Find the inlined wrapper group with translate(5,12)
    groups = [el for el in _iter_all(elements)
              if (el.tag.split("}")[-1] if "}" in el.tag else el.tag) == "g"
              and "translate(5,12)" in (el.get("transform") or "")]
    assert groups, "expected a <g transform='translate(5,12)'>"
    paths = [c for c in groups[0]
             if (c.tag.split("}")[-1] if "}" in c.tag else c.tag) == "path"]
    assert paths, "inlined group must contain the glyph path"


def test_zero_offset_no_transform() -> None:
    root = ET.fromstring(
        f'<svg xmlns="{NS}" xmlns:xlink="{XLINK}">'
        f'  <defs><g id="g0"><path id="p0" d="M 0 0"/></g></defs>'
        f'  <g><use xlink:href="#g0" x="0" y="0"/></g>'
        f'</svg>'
    )
    elements = list(root)
    _inline_latex_glyphs(elements)
    groups = [el for el in _iter_all(elements)
              if (el.tag.split("}")[-1] if "}" in el.tag else el.tag) == "g"
              and el.get("transform") is not None
              and "translate" in el.get("transform", "")]
    assert not groups, "zero-offset use must not produce a translate transform"


def test_noop_when_no_use() -> None:
    root = ET.fromstring(
        f'<svg xmlns="{NS}"><g><path id="standalone" d="M 0 0"/></g></svg>'
    )
    elements = list(root)
    _inline_latex_glyphs(elements)
    paths = [el for el in _iter_all(elements)
             if el.get("id") == "standalone"]
    assert paths, "standalone path must survive unchanged"


def _iter_all(elements: list[ET.Element]):
    for el in elements:
        yield el
        yield from el.iter()
