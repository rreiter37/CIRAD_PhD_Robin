#!/usr/bin/env python3
"""
Find the smallest subset of 'simple_preproc' such that, for every dataset,
at least one member of this subset appears in the Top-K (default K=3) Stage-1 performances.

This is a classic Minimum Set Cover / Hitting Set problem:
- Universe U = all datasets
- For each preprocessing p, define S_p = {datasets where p is in Top-K}
- Find the smallest collection of preprocessings whose union covers U.

We:
1) Filter Stage 1 rows only
2) For each (dataset, simple_preproc), keep the BEST Stage-1 metric (min)
3) Rank preprocessings within each dataset
4) Keep Top-K per dataset
5) Solve min cover:
   - Exact solver (branch & bound) with pruning
   - Greedy fallback/benchmark

Outputs:
- selected_simple_preproc_exact.csv (if exact solution found)
- selected_simple_preproc_greedy.csv
- coverage_report.csv
- Prints a clear console summary
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["dataset", "stage", "simple_preproc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the smallest subset of simple_preproc covering Top-K Stage-1 per dataset."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the TabPFN search results CSV (the one used by rank_simple_preproc_stage1.py).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to the parent directory of csv_path.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val_nrmse",
        choices=["val_nrmse", "val_rmse"],
        help="Stage-1 metric to use (lower is better). Default: val_nrmse.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Top-K preprocessings per dataset to consider. Default: 3.",
    )
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=200,
        help=(
            "Safety cap on the number of candidate preprocessings considered by the exact solver "
            "(kept by coverage size, descending). Default: 200."
        ),
    )
    parser.add_argument(
        "--no_exact",
        action="store_true",
        help="Disable exact solver and only use greedy.",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, metric: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if metric not in df.columns:
        missing.append(metric)
    if missing:
        raise ValueError(
            f"Missing required columns in CSV: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def compute_best_stage1(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Keep Stage 1 only, then for each (dataset, simple_preproc) keep the best metric (min).
    Returns columns: dataset, simple_preproc, best_metric, rank_in_dataset
    """
    df_s1 = df[df["stage"].astype(str).str.lower().eq("stage1")].copy()
    df_s1 = df_s1.dropna(subset=["dataset", "simple_preproc", metric])
    df_s1 = df_s1.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])

    if df_s1.empty:
        raise RuntimeError(
            "No Stage 1 rows found after filtering. "
            "Check that your CSV contains stage=='stage1' rows."
        )

    df_s1["simple_preproc"] = df_s1["simple_preproc"].astype(str)

    best = (
        df_s1.groupby(["dataset", "simple_preproc"], as_index=False)[metric]
        .min()
        .rename(columns={metric: f"best_{metric}"})
    )

    best["rank_in_dataset"] = (
        best.groupby("dataset")[f"best_{metric}"].rank(method="min", ascending=True)
    ).astype(float)

    return best


def build_topk_map(best: pd.DataFrame, metric: str, top_k: int) -> Dict[str, List[str]]:
    """
    Returns: topk_map[dataset] = list of simple_preproc in Top-K (ties included if rank<=K).
    """
    top = best[best["rank_in_dataset"] <= float(top_k)].copy()
    # Keep stable order: by rank then metric
    mcol = f"best_{metric}"
    top = top.sort_values(["dataset", "rank_in_dataset", mcol], ascending=[True, True, True])

    topk_map: Dict[str, List[str]] = {}
    for ds, g in top.groupby("dataset"):
        topk_map[str(ds)] = g["simple_preproc"].astype(str).tolist()

    return topk_map


def build_coverage_sets(topk_map: Dict[str, List[str]]) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    Universe U = datasets
    coverage[p] = set of datasets where p is in Top-K
    """
    datasets = set(topk_map.keys())
    coverage: Dict[str, Set[str]] = {}
    for ds, preprocs in topk_map.items():
        for p in preprocs:
            coverage.setdefault(p, set()).add(ds)
    return datasets, coverage


def greedy_set_cover(universe: Set[str], coverage: Dict[str, Set[str]]) -> List[str]:
    """
    Standard greedy approximation: repeatedly pick the preprocessing covering the most uncovered datasets.
    """
    uncovered = set(universe)
    selected: List[str] = []

    # Copy to avoid mutating input
    cov = {p: set(ds_set) for p, ds_set in coverage.items()}

    while uncovered:
        best_p = None
        best_gain = 0
        for p, ds_set in cov.items():
            gain = len(ds_set & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_p = p

        if best_p is None or best_gain == 0:
            # Cannot cover remaining datasets (should not happen if Top-K exists for all)
            break

        selected.append(best_p)
        uncovered -= cov[best_p]

    return selected


@dataclass(frozen=True)
class Candidate:
    name: str
    covers: frozenset


def exact_min_cover_branch_and_bound(universe: Set[str],
                                     coverage: Dict[str, Set[str]],
                                     max_candidates: int = 200) -> List[str] | None:
    """
    Exact minimum set cover via branch & bound.

    Practical tricks:
    - Restrict candidates to those that appear in at least one dataset Top-K (already true here)
    - Sort candidates by coverage size (descending)
    - Use lower bound via "max cover per remaining set" to prune
    - Branch on include/exclude
    """
    if not universe:
        return []

    # Build candidates and cap for safety (by coverage size)
    cands = [
        Candidate(name=p, covers=frozenset(ds_set))
        for p, ds_set in coverage.items()
        if len(ds_set) > 0
    ]
    cands.sort(key=lambda c: len(c.covers), reverse=True)
    cands = cands[:max_candidates]

    # Quick infeasibility check
    union_all = set().union(*(c.covers for c in cands)) if cands else set()
    if not universe.issubset(union_all):
        return None

    best_solution: List[str] | None = None

    # Precompute for pruning: list of candidate covers as python sets
    cand_sets = [set(c.covers) for c in cands]
    cand_names = [c.name for c in cands]

    # Helper: choose next uncovered dataset with smallest "options" (MRV heuristic)
    # We'll build an index: dataset -> list of candidate indices that cover it
    ds_to_cand_idxs: Dict[str, List[int]] = {ds: [] for ds in universe}
    for i, s in enumerate(cand_sets):
        for ds in s:
            if ds in ds_to_cand_idxs:
                ds_to_cand_idxs[ds].append(i)

    for ds in ds_to_cand_idxs:
        ds_to_cand_idxs[ds].sort()

    def lower_bound(uncovered: Set[str], start_idx: int) -> int:
        """
        Admissible lower bound: remaining uncovered / best possible per one pick among remaining candidates.
        """
        if not uncovered:
            return 0
        max_gain = 0
        for i in range(start_idx, len(cand_sets)):
            gain = len(cand_sets[i] & uncovered)
            if gain > max_gain:
                max_gain = gain
        if max_gain == 0:
            return 10**9
        return int(np.ceil(len(uncovered) / max_gain))

    def dfs(start_idx: int, uncovered: Set[str], chosen: List[int]) -> None:
        nonlocal best_solution

        # If covered all datasets, update best
        if not uncovered:
            sol = [cand_names[i] for i in chosen]
            if best_solution is None or len(sol) < len(best_solution):
                best_solution = sol
            return

        # Prune if already worse than best
        if best_solution is not None and len(chosen) >= len(best_solution):
            return

        # Bound
        lb = lower_bound(uncovered, start_idx)
        if best_solution is not None and (len(chosen) + lb) >= len(best_solution):
            return

        # Select a dataset to cover next (MRV: fewest remaining candidates cover it)
        # This improves branching efficiency.
        best_ds = None
        best_options = None
        for ds in uncovered:
            # candidates that cover ds AND are available from start_idx onward
            idxs = ds_to_cand_idxs.get(ds, [])
            options = [i for i in idxs if i >= start_idx]
            if not options:
                return  # infeasible in this branch
            if best_options is None or len(options) < len(best_options):
                best_options = options
                best_ds = ds
                if len(best_options) == 1:
                    break

        assert best_ds is not None and best_options is not None

        # Branch: try candidates that cover best_ds, in descending gain order
        # Order matters a lot for finding small solutions early.
        options = best_options
        options = sorted(options, key=lambda i: len(cand_sets[i] & uncovered), reverse=True)

        for i in options:
            # Include candidate i
            new_uncovered = uncovered - cand_sets[i]
            dfs(i + 1, new_uncovered, chosen + [i])

    dfs(0, set(universe), [])
    return best_solution


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = Path(args.out_dir) if args.out_dir is not None else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    metric = args.metric
    top_k = int(args.top_k)
    max_candidates = int(args.max_candidates)

    df = pd.read_csv(csv_path)
    validate_columns(df, metric)

    best = compute_best_stage1(df, metric=metric)
    topk_map = build_topk_map(best, metric=metric, top_k=top_k)

    # Sanity: ensure every dataset has at least one Top-K item
    missing_ds = [ds for ds, lst in topk_map.items() if len(lst) == 0]
    if missing_ds:
        raise RuntimeError(
            f"Some datasets have no Top-{top_k} items after filtering: {missing_ds}\n"
            f"Check your CSV or the 'stage'/'simple_preproc'/'{metric}' columns."
        )

    universe, coverage = build_coverage_sets(topk_map)

    # Greedy solution
    greedy = greedy_set_cover(universe, coverage)

    # Exact solution (optional)
    exact = None
    if not args.no_exact:
        exact = exact_min_cover_branch_and_bound(universe, coverage, max_candidates=max_candidates)

    # Save solutions
    greedy_path = out_dir / f"selected_simple_preproc_greedy_top{top_k}_stage1.csv"
    pd.DataFrame({"simple_preproc": greedy}).to_csv(greedy_path, index=False)

    exact_path = None
    if exact is not None:
        exact_path = out_dir / f"selected_simple_preproc_exact_top{top_k}_stage1.csv"
        pd.DataFrame({"simple_preproc": exact}).to_csv(exact_path, index=False)

    # Build coverage report (for the chosen solution; prefer exact if available)
    chosen = exact if exact is not None else greedy
    chosen_set = set(chosen)

    rows = []
    for ds in sorted(universe):
        top_list = topk_map[ds]
        hit = [p for p in top_list if p in chosen_set]
        rows.append(
            {
                "dataset": ds,
                f"top{top_k}_simple_preprocs": ";".join(top_list),
                "covered_by_subset": int(len(hit) > 0),
                "hits_in_topk": ";".join(hit),
            }
        )

    report = pd.DataFrame(rows)
    report_path = out_dir / f"coverage_report_top{top_k}_stage1.csv"
    report.to_csv(report_path, index=False)

    # Console summary
    n_ds = len(universe)
    print("\n" + "=" * 110)
    print(f"Minimum subset of simple_preproc covering Top-{top_k} Stage-1 per dataset")
    print("=" * 110)
    print(f"Input CSV:           {csv_path}")
    print(f"Output directory:    {out_dir}")
    print(f"Metric:              {metric} (lower is better)")
    print(f"Datasets (universe): {n_ds}")
    print(f"Candidates:          {len(coverage)} (preprocs appearing in at least one Top-{top_k})")
    print("-" * 110)
    print(f"Greedy subset size:  {len(greedy)}")
    print(f"Saved greedy →       {greedy_path}")

    if exact is None and not args.no_exact:
        print("Exact solver:        ❌ No exact solution found (or infeasible after candidate capping).")
        print(f"Tip: increase --max_candidates (current: {max_candidates}) or use --no_exact.")
    elif args.no_exact:
        print("Exact solver:        (disabled via --no_exact)")
    else:
        print(f"Exact subset size:   {len(exact)}")
        print(f"Saved exact →        {exact_path}")

    covered_rate = report["covered_by_subset"].mean() if not report.empty else 0.0
    print(f"Coverage rate:       {covered_rate * 100:.1f}%")
    print(f"Saved report →       {report_path}")

    print("\nSelected subset (used for report):")
    for i, p in enumerate(chosen, 1):
        print(f"  {i:02d}. {p}")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
