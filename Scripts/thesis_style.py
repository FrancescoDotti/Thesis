"""
thesis_style.py
Solarized Light matplotlib theme for all Thesis plots.
Call thesis_style.apply() before creating any figure.
"""
import matplotlib.pyplot as plt

SOLARIZED = {
    "base03":  "#002b36",
    "base02":  "#073642",
    "base01":  "#586e75",
    "base00":  "#657b83",
    "base0":   "#839496",
    "base1":   "#93a1a1",
    "base2":   "#eee8d5",
    "base3":   "#fdf6e3",
    "yellow":  "#b58900",
    "orange":  "#cb4b16",
    "red":     "#dc322f",
    "magenta": "#d33682",
    "violet":  "#6c71c4",
    "blue":    "#268bd2",
    "cyan":    "#2aa198",
    "green":   "#859900",
}

# Canonical colors for the three variance shares
SHARE_COLORS = {
    "MktInfoShare":  SOLARIZED["blue"],
    "FirmInfoShare": SOLARIZED["orange"],
    "NoiseShare":    SOLARIZED["green"],
}
SHARE_COLORS_LIST = [SOLARIZED["blue"], SOLARIZED["orange"], SOLARIZED["green"]]


def apply():
    """Apply Solarized Light theme to matplotlib globally."""
    plt.rcParams.update({
        "figure.facecolor":  SOLARIZED["base3"],
        "axes.facecolor":    SOLARIZED["base3"],
        "axes.edgecolor":    SOLARIZED["base1"],
        "axes.labelcolor":   SOLARIZED["base00"],
        "text.color":        SOLARIZED["base00"],
        "xtick.color":       SOLARIZED["base00"],
        "ytick.color":       SOLARIZED["base00"],
        "grid.color":        SOLARIZED["base2"],
        "axes.grid":         True,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "legend.facecolor":  SOLARIZED["base3"],
        "legend.edgecolor":  SOLARIZED["base1"],
        "savefig.facecolor": SOLARIZED["base3"],
    })
