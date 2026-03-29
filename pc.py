#!/usr/bin/env python3
"""Panel compiler: substitutes SVG figures into a panel template."""

import argparse
import copy
import logging
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

INKSCAPE_LABEL = "{http://www.inkscape.org/namespaces/inkscape}label"


class ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI color codes to log messages."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "",  # Default color
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        color = self.COLORS.get(levelname, self.RESET)
        record.levelname = f"{color}{levelname}{self.RESET}"
        return super().format(record)


class SVGDimensions:
    """SVG dimensions in source user units, for computing scale transforms."""

    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height

    @classmethod
    def from_svg(cls, svg_path: Path) -> "SVGDimensions":
        """Return the source user-unit extent of an SVG (viewBox dimensions).

        The scale transform ``scale(s)`` multiplies element coordinates by ``s``
        in the *target* SVG's user-unit space.  Physical units (pt, mm, …) on the
        ``width``/``height`` attributes are irrelevant for that ratio — only the
        viewBox extent, which matches element coordinate ranges, matters.
        """
        tree = ET.parse(svg_path)
        root = tree.getroot()

        viewbox = root.get("viewBox")
        if viewbox:
            parts = viewbox.split()
            return cls(width=float(parts[2]), height=float(parts[3]))

        # No viewBox: unitless width/height are the user units.
        width_attr = root.get("width")
        height_attr = root.get("height")
        if width_attr and height_attr:
            try:
                return cls(width=float(width_attr), height=float(height_attr))
            except ValueError:
                raise ValueError(
                    f"Cannot determine user-unit dimensions for {svg_path}: "
                    "no viewBox and physical-unit width/height are ambiguous"
                )

        raise ValueError(f"Cannot determine dimensions for {svg_path}")


def get_group_dimensions(
    group: ET.Element,
    config_dims: SVGDimensions | None = None,
) -> SVGDimensions | None:
    """Extract dimensions from group attributes, config, or bounding box."""
    # Try group attributes first
    width = group.get("width")
    height = group.get("height")

    if width and height:
        return SVGDimensions(width=float(width), height=float(height))

    # Fall back to config dimensions
    if config_dims:
        return config_dims

    # Try to calculate bounding box from children (e.g., placeholder rectangle)
    bbox = calculate_bbox(group)
    if bbox:
        return bbox

    # No dimensions available
    return None


def calculate_bbox(element: ET.Element) -> SVGDimensions | None:
    """Calculate bounding box from element and its children."""
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for elem in element.iter():
        # Extract x, y, width, height from common SVG elements
        x = elem.get("x")
        y = elem.get("y")
        w = elem.get("width")
        h = elem.get("height")

        if x and w:
            min_x = min(min_x, float(x))
            max_x = max(max_x, float(x) + float(w))
        if y and h:
            min_y = min(min_y, float(y))
            max_y = max(max_y, float(y) + float(h))

        # Also check cx, cy, r for circles
        cx = elem.get("cx")
        cy = elem.get("cy")
        r = elem.get("r")
        if cx and cy and r:
            cx_f, cy_f, r_f = float(cx), float(cy), float(r)
            min_x = min(min_x, cx_f - r_f)
            max_x = max(max_x, cx_f + r_f)
            min_y = min(min_y, cy_f - r_f)
            max_y = max(max_y, cy_f + r_f)

    if min_x != float("inf") and max_x != float("-inf"):
        width = max_x - min_x
        height = max_y - min_y if min_y != float("inf") else width
        if width > 0 and height > 0:
            return SVGDimensions(width=width, height=height)

    return None


def calculate_scale(
    source_dims: SVGDimensions,
    target_dims: SVGDimensions,
    fit: str,
) -> float:
    """Calculate scale factor based on fit strategy."""
    if fit == "height":
        return target_dims.height / source_dims.height
    elif fit == "width":
        return target_dims.width / source_dims.width
    elif fit == "contain":
        return min(
            target_dims.width / source_dims.width,
            target_dims.height / source_dims.height,
        )
    else:
        raise ValueError(f"Unknown fit option: {fit}")


def _rewrite_ids(elements: list[ET.Element], prefix: str) -> None:
    """Prefix all IDs and their references within elements to avoid conflicts."""
    # Collect all IDs first
    ids = set()
    for el in elements:
        for node in el.iter():
            if id_val := node.get("id"):
                ids.add(id_val)

    if not ids:
        return

    # Rewrite id attributes and references
    ref_attrs = {
        "href",
        "{http://www.w3.org/1999/xlink}href",
        "clip-path",
        "mask",
        "fill",
        "filter",
        "marker-start",
        "marker-mid",
        "marker-end",
    }
    for el in elements:
        for node in el.iter():
            if id_val := node.get("id"):
                node.set("id", f"{prefix}-{id_val}")
            for attr in ref_attrs:
                if val := node.get(attr):
                    for old_id in ids:
                        val = val.replace(f"#{old_id}", f"#{prefix}-{old_id}")
                    node.set(attr, val)
            # Also handle style attributes with url() references
            if style := node.get("style"):
                for old_id in ids:
                    style = style.replace(f"url(#{old_id})", f"url(#{prefix}-{old_id})")
                node.set("style", style)


def load_svg_content(svg_path: Path, id_prefix: str | None = None) -> list[ET.Element]:
    """Load SVG content as list of elements."""
    tree = ET.parse(svg_path)
    elements = [copy.deepcopy(element) for element in tree.getroot()]
    if id_prefix:
        _rewrite_ids(elements, id_prefix)
    return elements


def pdf_to_svg(pdf_path: Path) -> Path | None:
    """Convert PDF to SVG using inkscape, returning path to SVG in a temp dir.

    The caller is responsible for keeping the returned tempfile alive.
    Returns (svg_path, tmpdir) or None on failure.
    """
    tmpdir = tempfile.mkdtemp()
    svg_file = Path(tmpdir) / "out.svg"
    result = subprocess.run(
        ["pdf2svg", str(pdf_path), str(svg_file)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"pdf2svg failed for {pdf_path}\n{result.stderr}")
        return None
    return svg_file


def render_latex_to_svg(latex_text: str) -> list[ET.Element]:
    """Render LaTeX text to SVG using pdflatex and inkscape."""
    try:
        # Create temporary directory for LaTeX compilation
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create LaTeX document with standard size
            # We'll scale it in SVG afterwards
            latex_doc = f"""\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{geometry}}
\\geometry{{margin=5pt,paperwidth=500pt,paperheight=100pt}}
\\pagestyle{{empty}}
\\begin{{document}}
\\noindent
{latex_text}
\\end{{document}}
"""

            tex_file = tmpdir_path / "doc.tex"
            tex_file.write_text(latex_doc)

            # Compile with pdflatex
            result = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-output-directory",
                    str(tmpdir_path),
                    str(tex_file),
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error(
                    f"pdflatex failed for LaTeX: {latex_text}\n{result.stderr}"
                )
                return []

            # Convert PDF to SVG with Inkscape
            pdf_file = tmpdir_path / "doc.pdf"
            svg_file = tmpdir_path / "doc.svg"

            result = subprocess.run(
                [
                    "inkscape",
                    "--pdf-poppler",
                    str(pdf_file),
                    "--export-type=svg",
                    "--export-filename",
                    str(svg_file),
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error(
                    f"Inkscape conversion failed for LaTeX: {latex_text}\n{result.stderr}"
                )
                return []

            # Parse the generated SVG
            tree = ET.parse(svg_file)
            root = tree.getroot()

            # Extract all child elements from the root
            content = [copy.deepcopy(elem) for elem in root]

            logger.debug(f"Successfully rendered LaTeX: {latex_text}")
            return content

    except Exception as e:
        logger.error(
            f"Failed to render LaTeX text: {latex_text}\n"
            f"Error type: {type(e).__name__}\n"
            f"Error message: {e}"
        )
        return []


RESERVED_KEYS = {"panel", "output"}


def _write_output(tree: "ET.ElementTree[ET.Element]", output_path: Path) -> None:
    """Write a compiled SVG tree to an SVG or PDF output path."""
    if output_path.suffix.lower() == ".pdf":
        with tempfile.TemporaryDirectory() as tmpdir:
            svg_tmp = Path(tmpdir) / "out.svg"
            tree.write(svg_tmp, encoding="utf-8", xml_declaration=True)
            result = subprocess.run(
                [
                    "inkscape",
                    str(svg_tmp),
                    "--export-type=pdf",
                    "--export-filename",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error(f"Inkscape PDF export failed\n{result.stderr}")
                return
    else:
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
    logger.info(f"Panel compiled to {output_path}")


def _compile_one(panel_config: dict, config_path: Path, output_path: Path) -> None:
    """Compile a single panel from a config dict and write to one output path."""
    tree = _compile_tree(panel_config, config_path)
    if tree is not None:
        _write_output(tree, output_path)


def _compile_tree(
    panel_config: dict, config_path: Path
) -> "ET.ElementTree[ET.Element] | None":
    """Parse the template SVG, embed all figures, and return the compiled tree."""
    panel_str = panel_config.get("panel")
    if not panel_str:
        logger.error("Config missing required 'panel' key")
        return None
    panel_path = config_path.parent / panel_str

    if not panel_path.exists():
        logger.error(f"Panel file not found: {panel_path}")
        return None

    # Parse panel SVG
    tree = ET.parse(panel_path)
    root = tree.getroot()
    parent_map = {child: parent for parent in root.iter() for child in parent}

    # Register namespaces to preserve them in output
    namespaces = {
        "svg": "http://www.w3.org/2000/svg",
        "xlink": "http://www.w3.org/1999/xlink",
        "inkscape": "http://www.inkscape.org/namespaces/inkscape",
    }
    for prefix, uri in namespaces.items():
        ET.register_namespace(prefix, uri)

    # Process each figure in config
    for figure_id, figure_config in panel_config.items():
        if figure_id in RESERVED_KEYS:
            continue
        group = root.find(f".//*[@{INKSCAPE_LABEL}='{figure_id}']")
        if group is None:
            group = root.find(f".//*[@label='{figure_id}']")
        if group is None:
            group = root.find(f".//*[@id='{figure_id}']")
        if group is None:
            logger.warning(f"Group {figure_id} not found in panel")
            continue

        # Parse figure config
        tex_text = None
        fontsize = "10pt"
        if isinstance(figure_config, dict):
            svg_file = figure_config.get("file") or figure_config.get("svg")
            fit = figure_config.get("fit", "contain")
            config_width = figure_config.get("width")
            config_height = figure_config.get("height")
            tex_text = figure_config.get("tex")
            fontsize = figure_config.get("size", "10pt")
        else:
            svg_file = figure_config
            fit = "contain"
            config_width = None
            config_height = None

        # Extract numeric fontsize value
        try:
            if fontsize.endswith("pt"):
                fontsize_num = float(fontsize[:-2])
            else:
                fontsize_num = float(fontsize)
        except (ValueError, AttributeError):
            fontsize_num = 10.0

        # Use LaTeX if specified, otherwise require SVG file
        if tex_text:
            content = render_latex_to_svg(tex_text)
            if not content:
                logger.warning(f"Failed to render LaTeX for {figure_id}")
                continue
            # Scale LaTeX based on fontsize (12pt is base)
            scale = fontsize_num / 12.0
        elif svg_file:
            src_path = config_path.parent / svg_file
            if not src_path.exists():
                logger.warning(f"File not found: {src_path}")
                continue

            if src_path.suffix.lower() == ".pdf":
                svg_path = pdf_to_svg(src_path)
                if svg_path is None:
                    logger.warning(f"Failed to convert PDF for {figure_id}")
                    continue
            else:
                svg_path = src_path

            # Get dimensions from group or config
            config_dims = None
            if config_width and config_height:
                config_dims = SVGDimensions(
                    width=float(config_width), height=float(config_height)
                )

            source_dims = SVGDimensions.from_svg(svg_path)
            target_dims = get_group_dimensions(group, config_dims)

            # If no target dimensions, embed without scaling
            if target_dims is None:
                logger.debug(
                    f"Group {figure_id} has no width/height attributes and none specified in config. "
                    "Embedding without scaling."
                )
                scale = 1.0
            else:
                scale = calculate_scale(source_dims, target_dims, fit)

            content = load_svg_content(svg_path, id_prefix=figure_id)
        else:
            logger.warning(f"No SVG file or LaTeX text specified for {figure_id}")
            continue

        # Clear group and add scaled content
        original_attribs = dict(group.attrib)
        tag_name = group.tag.split("}")[-1] if "}" in group.tag else group.tag

        if tag_name == "rect":
            # Replace rect placeholder with a <g> translated to its x,y position.
            # SVG renderers do not render children of shape elements like <rect>.
            ns = group.tag.split("}")[0] + "}" if "}" in group.tag else ""
            container = ET.Element(f"{ns}g")
            container.set("id", figure_id)
            x = original_attribs.get("x", "0")
            y = original_attribs.get("y", "0")
            container.set("transform", f"translate({x},{y})")
            parent = parent_map[group]
            idx = list(parent).index(group)
            parent.remove(group)
            parent.insert(idx, container)
            group = container
        else:
            group.clear()
            group.attrib.update(original_attribs)

        for element in content:
            # Apply scale transform
            existing_transform = element.get("transform", "")
            scale_transform = f"scale({scale})" if scale != 1.0 else ""

            if existing_transform and scale_transform:
                element.set("transform", f"{scale_transform} {existing_transform}")
            elif scale_transform:
                element.set("transform", scale_transform)

            group.append(element)

    return tree


def compile_panel(config_path: Path, fallback_output: Path) -> None:
    """Compile one or more panels from a config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    def _resolve_outputs(raw: str | list[str] | None, fallback: Path) -> list[Path]:
        if not raw:
            return [fallback]
        items = raw if isinstance(raw, list) else [raw]
        return [config_path.parent / o for o in items]

    if isinstance(config, list):
        for item in config:
            raw = item.get("output")
            if not raw:
                logger.error("Each panel block must have an 'output' key")
                continue
            outputs = _resolve_outputs(raw, fallback_output)
            tree = _compile_tree(item, config_path)
            if tree is not None:
                for output_path in outputs:
                    _write_output(tree, output_path)
    else:
        outputs = _resolve_outputs(config.get("output"), fallback_output)
        tree = _compile_tree(config, config_path)
        if tree is not None:
            for output_path in outputs:
                _write_output(tree, output_path)


def main() -> None:
    """CLI entry point."""
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Compile SVG figures into a panel template"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('panel-compiler')}"
    )
    default_config = Path(__file__).stem + ".yaml"
    parser.add_argument(
        "config",
        nargs="?",
        default=default_config,
        help=f"Configuration YAML file (defaults to {default_config})",
    )
    args = parser.parse_args()

    for tool in ("inkscape", "pdf2svg"):
        if shutil.which(tool) is None:
            logger.warning(
                f"{tool} not found on PATH — PDF and LaTeX features will fail"
            )

    config_path = Path(args.config)
    output_path = config_path.with_suffix(".svg")

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return

    compile_panel(config_path, output_path)


if __name__ == "__main__":
    main()
