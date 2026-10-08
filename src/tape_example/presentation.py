"""Render the saved Figure 1 calculation and a completely offline demonstration."""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BLUE, RED = "#005A9C", "#B83C29"


def load_results(input_dir):
    source = Path(input_dir) / "continuum_results.json"
    return json.loads(source.read_text(encoding="utf-8")), source


def build_figure(data, font="STIXGeneral"):
    """Keep the original panel geometry and data. Choose the font explicitly."""
    resolved_font = font_manager.findfont(
        font_manager.FontProperties(family=font), fallback_to_default=False
    )
    plt.style.use(ROOT / "assets" / "sci_style.mplstyle")
    plt.rcParams.update({
        "font.family": font, "font.size": 9.5,
        "mathtext.fontset": "custom", "mathtext.rm": font,
        "mathtext.it": f"{font}:italic", "mathtext.bf": f"{font}:bold",
        "mathtext.default": "it", "mathtext.fallback": "stix", "axes.labelsize": 10,
        "axes.titlesize": 10.5, "axes.titleweight": "normal",
        "legend.fontsize": 8.6, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "text.color": "black", "axes.labelcolor": "black",
        "axes.edgecolor": "black", "xtick.color": "black",
        "ytick.color": "black", "legend.labelcolor": "black",
        "axes.linewidth": .75, "path.simplify": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.445, 3.18))
    fig.subplots_adjust(left=.205, right=.975, bottom=.185, top=.89)
    ax.set_title("Information after optimal design", pad=12)
    ax.plot(data["U"], data["ratio_upper"], color=BLUE, linewidth=1.65)
    grid = max(data["numerical_grid_screen"], key=lambda item: item["n"])
    rows = grid["rows"]
    xx = np.array([row["U"] for row in rows])
    yy = np.array([row["ratio"] for row in rows])
    keep = np.isin(np.round(xx, 5), [.1, .5, 1, 1.5, 2, 3, 4, 6])
    ax.plot(xx[keep], yy[keep], linestyle="none", marker="o", markersize=4.2,
            markerfacecolor="white", markeredgecolor=RED, markeredgewidth=.9)
    ax.set(xlim=(0, 6.1), ylim=(.296, .374),
           xlabel=r"Recovery window $U/T$",
           ylabel=r"Information ratio $J_C^\star/J_{CY,U}^\star$")
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(.02))
    ax.yaxis.set_minor_locator(MultipleLocator(.01))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.legend(handles=[
        Line2D([0], [0], color=BLUE, linewidth=1.65, label="Continuum upper bound"),
        Line2D([0], [0], linestyle="none", marker="o", markersize=4.2,
               markerfacecolor="white", markeredgecolor=RED, markeredgewidth=.9,
               label=f"{grid['n']} time blocks"),
    ], loc="upper right", frameon=False, handlelength=2.1, borderpad=0,
       labelspacing=.4)
    low, high = data["ratio_at_U2"]
    bracket = f"[{math.floor(low * 1000) / 1000:.3f}, {math.ceil(high * 1000) / 1000:.3f}]"
    ax.annotate(bracket, xy=(2, (low + high) / 2), xytext=(2.45, .327),
                fontsize=9.2, ha="left", va="center",
                arrowprops={"arrowstyle": "-", "color": "black", "linewidth": .6,
                            "shrinkA": 2, "shrinkB": 4})
    return fig, resolved_font


def plot(input_dir, output_dir, font="STIXGeneral"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, _ = load_results(input_dir)
    fig, _ = build_figure(data, font)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"figure1.{extension}", dpi=600)
    plt.close(fig)
    return output_dir / "figure1.png"


def demo(input_dir, output_dir, font="STIXGeneral"):
    output_dir = Path(output_dir)
    data, _ = load_results(input_dir)
    image = plot(input_dir, output_dir, font=font)
    grid = max(data["numerical_grid_screen"], key=lambda item: item["n"])
    cost = grid["cost"]["value"]
    points = [{"u": row["U"], "cost": cost, "tape": row["tape"]["value"],
               "ratio": cost / row["tape"]["value"],
               "multiple": row["tape"]["value"] / cost}
              for row in grid["rows"] if row["U"] > 0]
    payload = {"points": points, "n": grid["n"]}
    template = (ROOT / "assets" / "demo_template.html").read_text(encoding="utf-8")
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    content = template.replace("__FIGURE_PNG__", encoded).replace(
        "__RESULTS_JSON__", json.dumps(payload, ensure_ascii=True)
    )
    destination = output_dir / "demo.html"
    destination.write_text(content, encoding="utf-8")
    return destination
