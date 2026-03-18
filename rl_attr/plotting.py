from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "blue": "#1d4e89",
    "teal": "#0b6e69",
    "orange": "#d95f02",
    "gold": "#d9a404",
    "slate": "#495057",
    "red": "#b23a48",
    "light_blue": "#90caf9",
    "light_orange": "#f6bd60",
    "light_teal": "#84dcc6",
}


def configure_notebook_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "axes.prop_cycle": plt.cycler(
                color=[
                    PALETTE["blue"],
                    PALETTE["orange"],
                    PALETTE["teal"],
                    PALETTE["gold"],
                    PALETTE["red"],
                ]
            ),
        }
    )


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure_bundle(fig: plt.Figure, stem: str | Path) -> None:
    stem = Path(stem)
    ensure_directory(stem.parent)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(stem.with_suffix(suffix), facecolor="white")


def finalise_axes(ax: plt.Axes, *, xzero: bool = False, yzero: bool = False) -> None:
    if xzero:
        ax.axvline(0.0, color="#444444", linewidth=0.9, linestyle="--", alpha=0.8)
    if yzero:
        ax.axhline(0.0, color="#444444", linewidth=0.9, linestyle="--", alpha=0.8)


def annotate_curve_endpoints(ax: plt.Axes, x: Iterable[float], y: Iterable[float], label: str, color: str) -> None:
    x_array = np.asarray(list(x))
    y_array = np.asarray(list(y))
    ax.text(
        x_array[-1],
        y_array[-1],
        f"  {label}",
        color=color,
        va="center",
        ha="left",
        fontsize=10,
    )


def clean_bar_labels(ax: plt.Axes, rotation: int = 15) -> None:
    for tick in ax.get_xticklabels():
        tick.set_rotation(rotation)
        tick.set_horizontalalignment("right")
