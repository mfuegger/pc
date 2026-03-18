# pc — Panel Compiler

Compose scientific figures from SVG plots and LaTeX equations into a single SVG panel.

You define a panel template in Inkscape (or any SVG editor) with named group placeholders, then write a small YAML config that maps each placeholder to an SVG file or a LaTeX string. `pc` scales each figure to fit its placeholder and writes the compiled panel.

## Example

```
panel.svg  +  pc.yaml  →  out.svg
```

`panel.svg` — drawn in Inkscape, contains two empty groups `plot` and `label`:

```
pc.yaml`:
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
pip install panel-compiler
```

For LaTeX rendering, `pdflatex` and `inkscape` must be on your `PATH`.

## Usage

```
pc [config.yaml] [-o output.svg]
```

| Argument | Default | Description |
|---|---|---|
| `config.yaml` | `pc.yaml` | YAML config (see below) |
| `-o output.svg` | `out.svg` | Output file |

## Config format

Single panel:

```yaml
panel: path/to/panel.svg   # required — relative to this config file

# SVG figure
figure_id:
  file: path/to/plot.svg   # relative to this config file
  fit: contain             # contain | height | width  (default: contain)
  width: 200               # optional: override target dimensions
  height: 100

# LaTeX equation
equation_id:
  tex: $E = mc^2$          # any LaTeX math or text
  size: 12pt               # font size (default: 10pt)

# Shorthand — SVG file only
other_id: path/to/fig.svg
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

## License

Apache 2.0 — see [LICENSE](LICENSE).
