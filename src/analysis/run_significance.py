"""End-to-end runner: build the significance table for the conditions plotted in
notebooks/IVS_plot.ipynb. Saves CSV to outputs/significance_table.csv.

Run from the repo root:
    PYTHONPATH=. python src/analysis/run_significance.py
"""

from __future__ import annotations

import json
import os
import re

import pandas as pd

from src.analysis.significance import ConditionSpec, build_significance_table
from src.utils.data_utils import WVSAnalyzer

TARGET_COUNTRIES = ["Denmark", "Vietnam", "India", "Mexico"]

MODELS = [
    {
        "label": "Qwen3-4B-Instruct",
        "baseline": "qwen_prompt_steer_outputs_4/evaluation_results_default.json",
        "basic": "qwen_prompt_steer_outputs_4/evaluation_results_basic{country}.json",
        "advanced": "qwen_prompt_steer_outputs_new_prompt/evaluation_results_advanced_{country}_mtl.json",
        "binary_search_dir": "outputs_binary_search/Qwen_Qwen3-4B-Instruct-2507",
    },
    {
        "label": "Llama3.2-3B-Instruct",
        "baseline": "llama_prompt_steer_outputs/evaluation_results_default.json",
        "basic": "llama_prompt_steer_outputs/evaluation_results_basic{country}.json",
        "advanced": "llama_prompt_steer_outputs/evaluation_results_advanced_{country}_mtl.json",
        "binary_search_dir": "outputs_binary_search/meta-llama_Llama-3.2-3B-Instruct",
    },
    {
        "label": "Gemma3-4B-it",
        "baseline": "gemma_prompt_steer_outputs/evaluation_results_default.json",
        "basic": "gemma_prompt_steer_outputs/evaluation_results_basic{country}.json",
        "advanced": "gemma_prompt_steer_outputs/evaluation_results_advanced_{country}_mtl.json",
        "binary_search_dir": "outputs_binary_search/google_gemma-3-4b-it",
    },
]


def _binary_search_files(bs_dir: str) -> dict[tuple[str, str], str]:
    """Map (country, method_label) -> detail JSON path for best-coeff vector configs."""
    summary_path = os.path.join(bs_dir, "summary_results.json")
    with open(summary_path) as f:
        summary = json.load(f)
    out = {}
    for v in summary["vectors"]:
        if not v.get("is_best_coeff"):
            continue
        label = v["label"].replace(" (best_grid)", "")
        # label: "vector_{country}_vec_x_{coef}" or "vector_{country}_vec_x_advance_{coef}"
        m = re.match(r"vector_(\w+)_vec_x(?:_advance)?_([\-\d.eE]+)$", label)
        if not m:
            continue
        country = m.group(1)
        method = "Hybrid 2" if "_advance_" in label else "Hybrid 1"
        path = os.path.join(bs_dir, "details", f"{label}_mlt.json")
        if os.path.exists(path):
            out[(country, method)] = path
    return out


def build_conditions() -> list[ConditionSpec]:
    conds: list[ConditionSpec] = []
    for m in MODELS:
        bs = _binary_search_files(m["binary_search_dir"])
        for country in TARGET_COUNTRIES:
            conds.append(ConditionSpec(m["label"], country, "baseline", m["baseline"]))
            for method_name, tmpl in [("basic_prompt", m["basic"]), ("advanced_prompt", m["advanced"])]:
                p = tmpl.format(country=country)
                if os.path.exists(p):
                    conds.append(ConditionSpec(m["label"], country, method_name, p))
                else:
                    print(f"  missing: {p}")
            for method_name in ("vector", "vector+advanced"):
                p = bs.get((country, method_name))
                if p:
                    conds.append(ConditionSpec(m["label"], country, method_name, p))
                else:
                    print(f"  missing best-coeff vector for ({m['label']}, {country}, {method_name})")
    return conds


def main(n_boot: int = 10_000, out_csv: str = "outputs/significance_table.csv") -> pd.DataFrame:
    print("Loading WVSAnalyzer...")
    an = WVSAnalyzer()
    conds = build_conditions()
    print(f"Built {len(conds)} conditions")
    print("Running paired bootstrap...")
    df = build_significance_table(conds, an, n_boot=n_boot)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
