
import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_model_results(outputs_dir="outputs"):
    all_summaries = []
    files = glob.glob(f"{outputs_dir}/*/summary_results.json")
    
    for f in files:
        with open(f, 'r') as json_file:
            all_summaries.append(json.load(json_file))
    
    return all_summaries

def plot_cross_model_comparison(summaries):
    if not summaries:
        print("No results found in outputs/. Run run_pipeline.py first.")
        return

    # 1. Cultural Shift Distance Comparison
    # Euclidean distance from Baseline to Global Steering
    comparison_data = []
    for s in summaries:
        m_name = s['model_name']
        points = s['points']
        baseline = next(p for p in points if p['label'] == 'Baseline')
        steered = next(p for p in points if p['label'] == 'Global Steering')
        
        dist = np.sqrt((steered['RC1'] - baseline['RC1'])**2 + (steered['RC2'] - baseline['RC2'])**2)
        comparison_data.append({'Model': m_name, 'Cultural Shift Distance': dist})
    
    df_dist = pd.DataFrame(comparison_data)
    
    # 2. Perplexity Curve Comparison
    plt.figure(figsize=(12, 6))
    for s in summaries:
        ppl_data = s['perplexities']
        coeffs = sorted([float(c) for c in ppl_data.keys()])
        vals = [ppl_data[str(c)] for c in coeffs]
        plt.plot(coeffs, vals, marker='o', label=s['model_name'])
    
    plt.xlabel('Steering Coefficient')
    plt.ylabel('Perplexity')
    plt.title('Steering Cost (Fluency) Across Models')
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/paper_cross_model_perplexity.png")

    # 3. Bar Plot for Steerability
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_dist, x='Model', y='Cultural Shift Distance', palette='viridis')
    plt.title('Cultural Steerability Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/paper_cross_model_steerability.png")
    
    print("Cross-model comparison plots saved to outputs/")

if __name__ == "__main__":
    summaries = aggregate_model_results()
    plot_cross_model_comparison(summaries)
