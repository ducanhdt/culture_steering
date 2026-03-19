
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.analysis.plotting import (
    plot_cultural_map, plot_distance_deviation, 
    plot_perplexity_curve, plot_layer_steering_effect
)
from src.utils.data_utils import WVSAnalyzer
from src.core.config import DEFAULT_MODEL

def plot_saved_results(model_name=DEFAULT_MODEL):
    # 1. Load Data
    model_safe_name = model_name.replace("/", "_")
    output_dir = f"outputs/{model_safe_name}"
    summary_path = f"{output_dir}/summary_results.json"
    
    if not os.path.exists(summary_path):
        print(f"Error: No results found at {summary_path}. Run run_pipeline.py first.")
        return

    with open(summary_path, "r") as f:
        summary_data = json.load(f)
    
    analyzer = WVSAnalyzer()
    points_df = pd.DataFrame(summary_data.get("points", []))
    vectors_df = pd.DataFrame(summary_data.get("vectors", []))
    target_countries = summary_data.get("target_countries", [])
    
    # 2. Process Layer Data
    layer_diffs_raw = summary_data.get("layer_diffs", {})
    processed_diffs = {}
    processed_ids = {}
    if layer_diffs_raw:
        for q_id, diffs in layer_diffs_raw.items():
            top_4 = sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:4]
            processed_diffs[q_id] = [v for k, v in top_4]
            processed_ids[q_id] = [k for k, v in top_4]
    
    layer_diff_df = pd.DataFrame(processed_diffs)
    layer_id_df = pd.DataFrame(processed_ids)

    # 3. Setup Figure
    fig = plt.figure(figsize=(20, 15))
    
    # --- Subplot 1: Cultural Map ---
    ax1 = fig.add_subplot(2, 2, 1)
    if not points_df.empty:
        plot_cultural_map(ax1, points_df, wvs_data_df=analyzer.country_means, 
                          additional_vectors_df=vectors_df if not vectors_df.empty else None, 
                          title=f"Cultural Alignment ({model_name})")
    
    # --- Subplot 2: Distance Bar Chart ---
    ax2 = fig.add_subplot(2, 2, 2)
    if not points_df.empty and target_countries:
        target_means = analyzer.get_target_country_means()
        plot_distance_deviation(ax2, points_df, target_means, target_countries)

    # --- Subplot 3: Perplexity ---
    ax3 = fig.add_subplot(2, 2, 3)
    ppl_data = summary_data.get("perplexities", {})
    if ppl_data:
        coeffs = sorted([float(c) for c in ppl_data.keys()])
        perplexities = [ppl_data[str(c)] for c in coeffs]
        plot_perplexity_curve(ax3, coeffs, perplexities, label=model_name)

    # --- Subplot 4: Layer Analysis ---
    ax4 = fig.add_subplot(2, 2, 4)
    if not layer_diff_df.empty:
        plot_layer_steering_effect(ax4, layer_diff_df, layer_id_df)

    plt.tight_layout()
    plot_path = f"{output_dir}/visualizations_suite.png"
    plt.savefig(plot_path)
    plt.show()
    print(f"Visualization suite saved to {plot_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name used in experiments")
    args = parser.parse_args()
    
    plot_saved_results(args.model)
