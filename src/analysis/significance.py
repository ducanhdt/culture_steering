"""Bootstrap + Holm significance pipeline for cultural-map steering results.

Consumes the per-item detail JSONs already produced by the evaluation pipeline
(each file has 300 records with `wvs_id`, `human_aligned_score`, etc.) and
produces a paired-bootstrap p-value for the hypothesis:

    distance(method, target_country) < distance(baseline, target_country)

within each (model, country). Holm correction is applied across the
(model x method) family inside each country.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from src.utils.data_utils import WVSAnalyzer

IV_QNS = ["A008", "A165", "E018", "E025", "F063", "F118", "F120", "G006", "Y002", "Y003"]


def _load_items(path: str) -> pd.DataFrame:
    with open(path) as f:
        items = json.load(f)
    df = pd.DataFrame(items)
    missing = [q for q in IV_QNS if q not in df["wvs_id"].unique()]
    if missing:
        raise ValueError(f"{path}: missing wvs_ids {missing}")
    return df


def _project_means(items: pd.DataFrame, indices: np.ndarray, analyzer: WVSAnalyzer) -> tuple[np.ndarray, np.ndarray]:
    """Given per-iter sampled-row indices [N, K], compute (RC1, RC2) per iter.

    `indices` shape: (n_boot, n_items). For each row, group sampled items by
    wvs_id, take the mean of human_aligned_score, then project to RC1/RC2.
    """
    wvs_ids = items["wvs_id"].to_numpy()
    scores = items["human_aligned_score"].to_numpy(dtype=float)

    qn_to_idx = {q: i for i, q in enumerate(IV_QNS)}
    qn_codes = np.array([qn_to_idx[q] for q in wvs_ids])

    n_boot, k = indices.shape
    n_qns = len(IV_QNS)

    sums = np.zeros((n_boot, n_qns))
    counts = np.zeros((n_boot, n_qns), dtype=np.int64)

    sampled_codes = qn_codes[indices]
    sampled_scores = scores[indices]

    for q in range(n_qns):
        mask = sampled_codes == q
        sums[:, q] = np.where(mask, sampled_scores, 0.0).sum(axis=1)
        counts[:, q] = mask.sum(axis=1)

    if (counts == 0).any():
        # extremely unlikely with 30 items/question, but guard anyway
        bad = (counts == 0).sum()
        raise RuntimeError(f"{bad} bootstrap iterations had an empty wvs_id bucket")

    means = sums / counts
    means_df = pd.DataFrame(means, columns=IV_QNS)
    rc1, rc2 = analyzer.project_scores(means_df)
    return rc1, rc2


def bootstrap_condition(
    detail_json_path: str,
    analyzer: WVSAnalyzer,
    n_boot: int = 10_000,
    seed: int = 0,
    indices: np.ndarray | None = None,
) -> dict:
    """Bootstrap (RC1, RC2) for a single condition.

    If `indices` is provided, use those instead of resampling — for paired
    bootstrap.
    """
    items = _load_items(detail_json_path)
    n = len(items)

    if indices is None:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, n, size=(n_boot, n))
    else:
        if indices.shape[1] != n:
            raise ValueError(
                f"indices width {indices.shape[1]} != #items {n} in {detail_json_path}"
            )

    rc1, rc2 = _project_means(items, indices, analyzer)
    return {"rc1": rc1, "rc2": rc2, "indices": indices, "items": items}


def _items_aligned(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if len(a) != len(b):
        return False
    return (
        a["wvs_id"].equals(b["wvs_id"].reset_index(drop=True))
        and a["scenario_text"].equals(b["scenario_text"].reset_index(drop=True))
    )


def paired_bootstrap_diff(
    baseline_path: str,
    method_path: str,
    target_rc1: float,
    target_rc2: float,
    analyzer: WVSAnalyzer,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    """Bootstrap distribution of Δdistance = dist(baseline) - dist(method).

    Pairs items by row order if `wvs_id` and `scenario_text` align between
    files; otherwise runs independent resamples and warns.
    """
    base_items = _load_items(baseline_path)
    meth_items = _load_items(method_path)

    aligned = _items_aligned(base_items, meth_items)
    rng = np.random.default_rng(seed)

    if aligned:
        idx = rng.integers(0, len(base_items), size=(n_boot, len(base_items)))
        base = bootstrap_condition(baseline_path, analyzer, n_boot, seed, indices=idx)
        meth = bootstrap_condition(method_path, analyzer, n_boot, seed, indices=idx)
        pairing = "paired"
    else:
        base = bootstrap_condition(baseline_path, analyzer, n_boot, seed)
        meth = bootstrap_condition(method_path, analyzer, n_boot, seed + 1)
        pairing = "unpaired"

    dist_base = np.hypot(base["rc1"] - target_rc1, base["rc2"] - target_rc2)
    dist_meth = np.hypot(meth["rc1"] - target_rc1, meth["rc2"] - target_rc2)
    delta = dist_base - dist_meth  # >0 means method is closer than baseline

    n_neg_or_zero = int((delta <= 0).sum())
    p_one_sided = (1 + n_neg_or_zero) / (n_boot + 1)
    ci_low, ci_high = np.quantile(delta, [0.025, 0.975])

    return {
        "pairing": pairing,
        "delta": delta,
        "dist_baseline": dist_base,
        "dist_method": dist_meth,
        "rc1_baseline": base["rc1"],
        "rc2_baseline": base["rc2"],
        "rc1_method": meth["rc1"],
        "rc2_method": meth["rc2"],
        "delta_mean": float(delta.mean()),
        "delta_ci_low": float(ci_low),
        "delta_ci_high": float(ci_high),
        "dist_baseline_mean": float(dist_base.mean()),
        "dist_method_mean": float(dist_meth.mean()),
        "p_raw": float(p_one_sided),
    }


def holm_correct(pvals: Iterable[float]) -> np.ndarray:
    pvals = np.asarray(list(pvals), dtype=float)
    if len(pvals) == 0:
        return pvals
    _, padj, _, _ = multipletests(pvals, method="holm")
    return padj


@dataclass
class ConditionSpec:
    model: str
    country: str
    method: str  # "baseline" reserved for the per-model default
    path: str


def _target_coords(analyzer: WVSAnalyzer, country: str) -> tuple[float, float]:
    cm = analyzer.country_means
    row = cm[cm["country.territory"] == country]
    if row.empty:
        raise KeyError(f"country {country!r} not in WVSAnalyzer.country_means")
    return float(row.iloc[0]["RC1_final"]), float(row.iloc[0]["RC2_final"])


def build_significance_table(
    conditions: list[ConditionSpec],
    analyzer: WVSAnalyzer,
    n_boot: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Run paired bootstrap for every non-baseline (model, country, method).

    Holm correction is applied within each country across all (model x method).
    """
    by_mc_baseline = {}
    methods = []
    for c in conditions:
        if c.method == "baseline":
            by_mc_baseline[(c.model, c.country)] = c.path
        else:
            methods.append(c)

    rows = []
    for c in methods:
        baseline_path = by_mc_baseline.get((c.model, c.country))
        if baseline_path is None:
            raise KeyError(f"no baseline for ({c.model}, {c.country})")
        t_rc1, t_rc2 = _target_coords(analyzer, c.country)
        res = paired_bootstrap_diff(
            baseline_path, c.path, t_rc1, t_rc2, analyzer,
            n_boot=n_boot, seed=seed,
        )
        rows.append({
            "model": c.model,
            "country": c.country,
            "method": c.method,
            "pairing": res["pairing"],
            "dist_method_mean": res["dist_method_mean"],
            "dist_baseline_mean": res["dist_baseline_mean"],
            "delta_mean": res["delta_mean"],
            "delta_ci_low": res["delta_ci_low"],
            "delta_ci_high": res["delta_ci_high"],
            "p_raw": res["p_raw"],
        })

    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for country, sub in df.groupby("country"):
        df.loc[sub.index, "p_holm"] = holm_correct(sub["p_raw"].values)
    df["significant"] = df["p_holm"] < 0.05
    return df.sort_values(["country", "model", "method"]).reset_index(drop=True)
