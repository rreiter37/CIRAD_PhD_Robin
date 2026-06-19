#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_tabpfn_accuracy_reliability.py

Render panel (b) of the TabPFN-Raw vs TabPFN-opt figure (manuscript_ACA,
"Influence of preprocessing"): an aggregated *reliability diagram* of the native
TabPFN predictive distribution.

For each model variant, the curve shows the mean empirical coverage of the
central prediction interval (across the regression datasets) as a function of
the nominal coverage level. The diagonal is perfect calibration; a curve above
the diagonal indicates under-confidence (intervals too wide), below indicates
over-confidence (intervals too narrow). The shaded band is the inter-quartile
range across datasets.

Panel (a) of the final figure is the existing per-dataset iRMSEP dumbbell
(``dumbbell_irmsep_vs_pls_size_outliers_removed_tabpfn.png``); the two panels are
composed in the manuscript with a LaTeX subfigure environment. The aggregated
sharpness (NMPIW) numbers reported in the caption come from
``uncertainty_summary.json``.

Input : ``reliability_curve.csv`` produced by ``compute_uncertainty_tabpfn.py``.
Output: a 300 dpi PNG of the reliability diagram.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

#: Consistent colors for the two variants (kept close to the rest of the paper).
MODEL_STYLE: Dict[str, Dict[str, str]] = {
    "TabPFN-Raw": {"color": "#4C72B0", "label": "TabPFN-Raw"},
    "TabPFN-opt": {"color": "#C44E52", "label": "TabPFN-opt"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curve", type=str,
                   default="Results/tabpfn_uncertainty/reliability_curve.csv",
                   help="Path to reliability_curve.csv.")
    p.add_argument("--summary", type=str,
                   default="Results/tabpfn_uncertainty/uncertainty_summary.json",
                   help="Path to uncertainty_summary.json (for the caption note).")
    p.add_argument("--outfile", type=str,
                   default="Article_TabPFN/Figures/reliability_tabpfn_raw_vs_opt.png",
                   help="Output PNG path (300 dpi).")
    p.add_argument("--band", choices=["iqr", "none"], default="iqr",
                   help="Show the inter-quartile band across datasets.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    curve = pd.read_csv(args.curve)

    fig, ax = plt.subplots(figsize=(4.6, 4.4))

    # Perfect-calibration diagonal.
    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="0.4", zorder=1,
            label="Perfect calibration")

    for model, style in MODEL_STYLE.items():
        g = curve[curve["model"] == model].sort_values("nominal")
        if g.empty:
            continue
        ax.plot(g["nominal"], g["mean"], "-o", ms=3.5, lw=1.6,
                color=style["color"], label=style["label"], zorder=3)
        if args.band == "iqr" and {"q25", "q75"}.issubset(g.columns):
            ax.fill_between(g["nominal"], g["q25"], g["q75"],
                            color=style["color"], alpha=0.15, zorder=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Reliability of the predictive distribution")
    ax.text(0.97, 0.06, "under-confident\n(above diagonal)", ha="right", va="bottom",
            fontsize=7, color="0.4", transform=ax.transAxes)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(True, ls=":", lw=0.5, alpha=0.5)

    # Optional caption helper: print sharpness numbers if available.
    summ_path = Path(args.summary)
    if summ_path.exists():
        summ = json.loads(summ_path.read_text())
        meds = summ.get("per_model_median", {})
        note = " | ".join(
            f"{m}: NMPIW@90={v.get('nmpiw@90', float('nan')):.2f},"
            f" nCRPS={v.get('crps_norm', float('nan')):.3f}"
            for m, v in meds.items()
        )
        if note:
            print("Caption sharpness note:", note)

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved reliability diagram -> {out}")


if __name__ == "__main__":
    main()
