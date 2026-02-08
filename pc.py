#!/usr/bin/env python3
"""Panel compiler: substitutes SVG figures into a panel template."""

import argparse
import copy
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class SVGDimensions:
    """SVG dimensions extracted from viewBox or width/height."""

    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height

    @classmethod
    def from_svg(cls, svg_path: Path) -> "SVGDimensions":
        """Extract dimensions from SVG file."""
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # Try viewBox first (more reliable for scaled SVGs)
        viewbox = root.get("viewBox")
        if viewbox:
            parts = viewbox.split()
            return cls(width=float(parts[2]), height=float(parts[3]))

        # Fall back to width/height attributes
        width = root.get("width")
        height = root.get("height")
        if width and height:
            return cls(width=float(width), height=float(height))

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


def load_svg_content(svg_path: Path) -> list[ET.Element]:
    """Load SVG content as list of elements."""
    tree = ET.parse(svg_path)
    # Return deep copies to avoid element reuse when multiple groups use the same SVG
    return [copy.deepcopy(element) for element in tree.getroot()]


def compile_panel(
    panel_path: Path,
    config_path: Path,
    output_path: Path,
) -> None:
    """Compile panel by substituting figures from config."""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Parse panel SVG
    tree = ET.parse(panel_path)
    root = tree.getroot()

    # Register namespaces to preserve them in output
    namespaces = {
        "svg": "http://www.w3.org/2000/svg",
        "xlink": "http://www.w3.org/1999/xlink",
    }
    for prefix, uri in namespaces.items():
        ET.register_namespace(prefix, uri)

    # Process each figure in config
    for figure_id, figure_config in config.items():
        group = root.find(f".//*[@id='{figure_id}']")
        if group is None:
            logger.warning(f"Group {figure_id} not found in panel")
            continue

        # Parse figure config
        if isinstance(figure_config, dict):
            svg_file = figure_config.get("file") or figure_config.get("svg")
            fit = figure_config.get("fit", "contain")
            config_width = figure_config.get("width")
            config_height = figure_config.get("height")
        else:
            svg_file = figure_config
            fit = "contain"
            config_width = None
            config_height = None

        if not svg_file:
            logger.warning(f"No SVG file specified for {figure_id}")
            continue

        svg_path = panel_path.parent / svg_file
        if not svg_path.exists():
            logger.warning(f"SVG file not found: {svg_path}")
            continue

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
            logger.warning(
                f"Group {figure_id} has no width/height attributes and none specified in config. "
                "Embedding without scaling."
            )
            scale = 1.0
        else:
            scale = calculate_scale(source_dims, target_dims, fit)

        # Load source SVG content
        content = load_svg_content(svg_path)

        # Clear group and add scaled content
        # Preserve all original attributes
        original_attribs = dict(group.attrib)
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

    # Write output
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    logger.info(f"Panel compiled to {output_path}")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Compile SVG figures into a panel template"
    )
    parser.add_argument("panel", help="Input panel SVG file")
    parser.add_argument(
        "config",
        nargs="?",
        default="pc.yaml",
        help="Configuration YAML file (defaults to pc.yaml in current directory)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output SVG file (defaults to overwriting input panel)",
    )

    args = parser.parse_args()

    panel_path = Path(args.panel)
    config_path = Path(args.config)
    output_path = Path(args.output) if args.output else panel_path

    if not panel_path.exists():
        logger.error(f"Panel file not found: {panel_path}")
        return

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return

    compile_panel(panel_path, config_path, output_path)


if __name__ == "__main__":
    main()
