# pc — Panel Compiler

Compose scientific figures from SVG plots, PDF figures, and LaTeX equations into a single SVG or PDF panel.

You define a panel template in Inkscape with named group placeholders (set via the layer/group label field), then write a small YAML config that maps each label to an SVG file, PDF, or a LaTeX string. `pc` scales each figure to fit its placeholder and writes the compiled panel.

## Example

```
panel.svg  +  pc.yaml  →  out.svg
```

`panel.svg` — drawn in Inkscape, contains two empty groups `plot` and `label`.

`pc.yaml`:
```yaml
panel: panel.svg

plot:
  file: results.svg
  fit: contain

label:
  tex: $y = \sin(x) + \cos(x)$
  size: 12pt
```

```bash
pc pc.yaml -o out.svg
```

## Installation

```bash
pip install git+https://github.com/mfuegger/pc.git
# or with uv
uv tool install git+https://github.com/mfuegger/pc.git
```

For LaTeX rendering, `pdflatex` must be on your `PATH`. For PDF figures and PDF output, `pdf2svg` and `inkscape` must be on your `PATH`. On macOS: `brew install pdf2svg`.

## Usage

```
pc [config.yaml] [-o output.svg|pdf]
```

| Argument | Default | Description |
|---|---|---|
| `config.yaml` | `pc.yaml` | YAML config (see below) |
| `-o output` | `out.svg` | Output file — `.svg` or `.pdf` |

## Config format

Single panel — `output` accepts a string or a list to produce multiple formats at once:

```yaml
panel: path/to/panel.svg   # required — relative to this config file

# SVG or PDF figure
figure_id:
  file: path/to/plot.svg   # .svg or .pdf — relative to this config file
  fit: contain             # contain | height | width  (default: contain)
  width: 200               # optional: override target dimensions
  height: 100

# LaTeX equation
equation_id:
  tex: $E = mc^2$          # any LaTeX math or text
  size: 12pt               # font size (default: 10pt)

# Shorthand — SVG file only
other_id: path/to/fig.svg

# Single output (string)
output: out.svg

# Multiple outputs — produces both files from one run
output:
  - out.svg
  - out.pdf
```

Multiple panels in one config (list form) — `-o` is ignored, each block needs `output`:

```yaml
- panel: panel1.svg
  output: out1.svg
  plot:
    file: results.svg

- panel: panel2.svg
  output: out2.svg
  label:
    tex: $E = mc^2$
    size: 12pt
```

**Fit strategies:**

- `contain` — scale to fit within the placeholder (default)
- `height` — scale to match placeholder height
- `width` — scale to match placeholder width

Placeholder dimensions are read from the group element's `width`/`height` attributes in the panel SVG. You can override them in the config.

**Group matching** uses `inkscape:label` first (the label field in Inkscape's XML editor or layers panel), then plain `label`, then `id`.

## License

Apache 2.0 — see [LICENSE](LICENSE).
