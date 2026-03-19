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
pc pc.yaml
```

## Installation

```bash
uv tool install git+https://github.com/mfuegger/pc.git
```

or with pip:

```bash
pip install git+https://github.com/mfuegger/pc.git
```

For LaTeX rendering, `pdflatex` must be on your `PATH`. For PDF figures and PDF output, `pdf2svg` and `inkscape` must be on your `PATH`. On macOS: `brew install pdf2svg`.

## Usage

```
pc [config.yaml]
```

| Argument | Default | Description |
|---|---|---|
| `config.yaml` | `pc.yaml` | YAML config (see below) |

The output path is taken from the `output` key in the config. If omitted, it defaults to the config filename with `.svg` extension (e.g. `pc.yaml` → `pc.svg`).

## Config format

### Single panel

```yaml
panel: panel.svg    # required — path to the panel SVG, relative to this config file
output: out.svg     # optional — defaults to <config-stem>.svg if omitted
```

`output` can also be a list to produce multiple formats in one run:

```yaml
output:
  - out.svg
  - out.pdf
```

Each remaining key is a **group label** that must match a group in the panel SVG. `pc` looks up the group by `inkscape:label` first, then `label`, then `id`.

### Figure entry (SVG or PDF)

```yaml
plot:
  file: results.svg   # path relative to this config file; .svg or .pdf  (alias: svg:)
  fit: contain        # contain | height | width  (default: contain)
  width: 200          # optional — override the target width  (SVG user units)
  height: 100         # optional — override the target height (SVG user units)
```

Fit strategies:
- `contain` — scale uniformly to fit within the placeholder box (default)
- `height` — scale to match the placeholder height exactly
- `width`  — scale to match the placeholder width exactly

Target dimensions come from `width`/`height` attributes on the group element in the panel SVG. The config `width`/`height` are used as a fallback when the group has no such attributes.

PDF files are converted via `pdf2svg` before embedding.

### LaTeX entry

```yaml
label:
  tex: $y = \sin(x)$   # any LaTeX — math mode, text, amsmath, …
  size: 12pt           # font size (default: 10pt); scales the rendered output
```

Rendered via `pdflatex` + `inkscape`. No fit scaling — the output is sized by `size` alone.

### Shorthand

A plain string value is treated as an SVG file path with `fit: contain`:

```yaml
other: path/to/fig.svg
```

### Multiple panels

Use a YAML list; each entry must have its own `output`:

```yaml
- panel: panel1.svg
  output: out1.svg
  plot:
    file: results.svg

- panel: panel2.svg
  output:
    - out2.svg
    - out2.pdf
  label:
    tex: $E = mc^2$
    size: 12pt
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
