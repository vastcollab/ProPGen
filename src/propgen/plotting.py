"""Shared matplotlib styling for the figure notebooks.

The published figures use Helvetica Neue. That font is generally absent on
Linux and in continuous integration, where requesting it produces a stream of
warnings and a silent substitution. :func:`set_paper_style` picks the best
available font from a preference list instead, so notebooks render cleanly
everywhere while still matching the paper wherever the font is installed.
"""

from __future__ import annotations

__all__ = ["set_paper_style", "GP_COLORS"]

#: Colours used for genotype-phenotype pairs in the paper's trajectory panels.
GP_COLORS = {
    (0, 0): "tab:blue",
    (0, 1): "tab:orange",
    (1, 0): "tab:green",
    (1, 1): "tab:red",
    (2, 0): "tab:purple",
    (2, 1): "tab:brown",
}

_FONT_PREFERENCE = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def set_paper_style(font_size: int = 20) -> str:
    """Apply the paper's matplotlib style. Returns the font actually used."""
    import matplotlib as mpl
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in _FONT_PREFERENCE if f in available), "DejaVu Sans")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [chosen, *_FONT_PREFERENCE],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size * 0.8,
            "ytick.labelsize": font_size * 0.8,
            "legend.fontsize": font_size * 0.7,
            "mathtext.fontset": "custom",
            "mathtext.rm": chosen,
            "mathtext.it": f"{chosen}:italic",
            "mathtext.bf": f"{chosen}:bold",
            # Without these, the custom fontset falls back to the 'cursive' and
            # 'monospace' generic families and warns on every render.
            "mathtext.cal": f"{chosen}:italic",
            "mathtext.sf": chosen,
            "mathtext.tt": "DejaVu Sans Mono",
            "figure.dpi": 100,
            "savefig.bbox": "tight",
            "savefig.transparent": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return chosen
