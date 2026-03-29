import xml.etree.ElementTree as ET

from pc import _rewrite_ids


def test_prefixes_id() -> None:
    elem = ET.fromstring('<g xmlns="http://www.w3.org/2000/svg" id="foo"/>')
    _rewrite_ids([elem], "pfx")
    assert elem.get("id") == "pfx-foo"


def test_updates_href() -> None:
    elem = ET.fromstring(
        '<g xmlns="http://www.w3.org/2000/svg"><use href="#foo"/><g id="foo"/></g>'
    )
    _rewrite_ids([elem], "pfx")
    use = elem.find("{http://www.w3.org/2000/svg}use")
    assert use is not None
    assert use.get("href") == "#pfx-foo"


def test_updates_style_url() -> None:
    elem = ET.fromstring(
        '<g xmlns="http://www.w3.org/2000/svg">'
        '<path id="clip1"/>'
        '<rect style="clip-path: url(#clip1)"/>'
        "</g>"
    )
    _rewrite_ids([elem], "fig")
    rect = elem.find("{http://www.w3.org/2000/svg}rect")
    assert rect is not None
    assert "url(#fig-clip1)" in (rect.get("style") or "")


def test_no_ids_is_noop() -> None:
    elem = ET.fromstring('<g xmlns="http://www.w3.org/2000/svg"><rect/></g>')
    _rewrite_ids([elem], "pfx")  # must not raise
