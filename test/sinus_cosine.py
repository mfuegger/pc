#!/usr/bin/env python3
"""Generate a sine + cosine wave SVG plot."""

import numpy as np
import matplotlib.pyplot as plt


def generate_trig_plot(output_path: str = "sinus.svg") -> None:
    """Generate sine and cosine plot and save as SVG."""
    x = np.linspace(0, 4 * np.pi, 200)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y_sin, linewidth=2, label="sin(x)", color="steelblue")
    ax.plot(x, y_cos, linewidth=2, label="cos(x)", color="coral")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Sine and Cosine Waves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 1.5)

    fig.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"Saved trig plot to {output_path}")


def generate_cosine_plot(output_path: str = "cos.svg") -> None:
    """Generate cosine-only plot and save as SVG."""
    x = np.linspace(0, 4 * np.pi, 200)
    y_cos = np.cos(x)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y_cos, linewidth=2, label="cos(x)", color="coral")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Cosine Wave")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 1.5)

    fig.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"Saved cosine plot to {output_path}")


if __name__ == "__main__":
    generate_trig_plot()
    generate_cosine_plot()
