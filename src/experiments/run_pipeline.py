
import json
import os
import pandas as pd
import numpy as np
import gc
import torch
from src.core.evaluator import CulturalEvaluator
from src.core.trainer import train_cultural_vector
from src.core.config import (
    DEFAULT_MODEL, TARGET_COUNTRIES, 
    BASIC_PROMPT_TEMPLATE, ADVANCE_PROMPTS_MLT, ADVANCE_PROMPTS
)
from src.utils.data_utils import WVSAnalyzer

def save_summary(output_dir, summary_data):
    with open(f"{output_dir}/summary_results.json", "w") as f:
        json.dump(summary_data, f, indent=4)

def save_detailed(output_dir, name, results):
    details_dir = f"{output_dir}/details"
    os.makedirs(details_dir, exist_ok=True)
    with open(f"{details_dir}/{name}.json", "w") as f:
        json.dump(results, f, indent=4)

def release_memory(force=False):
    gc.collect()
    if force and torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_paper_experiments(model_name=DEFAULT_MODEL,
                         train_path="data/train_data_mtl.json",
                         test_path="data/sample_data_mtl.json",
                         questions_path="data/culture_questions.json",
                         best_layer_ids=None,
                         test=False):
    
    # 1. Setup
    model_safe_name = model_name.replace("/", "_")
    output_dir = f"outputs/{model_safe_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = WVSAnalyzer()
    
    with open(train_path, 'r') as f: train_data = json.load(f)
    with open(test_path, 'r') as f: test_data = json.load(f)
    with open(questions_path, 'r') as f: 
        questions = json.load(f)
    id_to_info = {item["ID"]: item for item in questions}
    
    evaluator = CulturalEvaluator(model_name, id_to_info=id_to_info)
    
    # Use only first country if test mode is enabled
    if test:
        print("TEST MODE: Using only the first target country for faster execution.")
        countries_to_use = TARGET_COUNTRIES[:2] 
        test_data = test_data[:100]  # Use a subset of test data for faster evaluation 
        train_data = train_data[:60]
    else:
        countries_to_use = TARGET_COUNTRIES
    summary_data = {
        "model_name": model_name,
        "target_countries": countries_to_use,
        "points": [],
        "vectors": [],
        "perplexities": {},
        "layer_diffs": {},
        "domain_shifts": {}
    }
    
    # --- STEP 1: BASELINE ---
    print("Evaluating Baseline...")
    res_baseline = evaluator.evaluate_dataset(test_data)
    save_detailed(output_dir, "baseline", res_baseline)
    scores_baseline = evaluator.aggregate_cultural_scores(res_baseline, analyzer=analyzer)
    summary_data["points"].append({
        'RC1': float(scores_baseline['X_Axis']), 
        'RC2': float(scores_baseline['Y_Axis']), 
        'label': 'Baseline', 
        'color': 'red'
    })
    release_memory(force=True)

    # --- STEP 2: PROMPT STEERING ---
    # for country in countries_to_use:
    #     print(f"[{country}] Running Prompt Interventions...")
    #     # Basic
    #     res_basic = evaluator.evaluate_dataset(test_data, system_prompt=BASIC_PROMPT_TEMPLATE.format(country=country))
    #     save_detailed(output_dir, f"basic_{country}", res_basic)
    #     s_basic = evaluator.aggregate_cultural_scores(res_basic, analyzer=analyzer)
    #     summary_data["points"].append({
    #         'RC1': float(s_basic['X_Axis']), 'RC2': float(s_basic['Y_Axis']), 
    #         'label': f'Basic: {country}', 'color': 'purple'
    #     })
        
    #     # Advanced (English)
    #     res_adv_en = evaluator.evaluate_dataset(test_data, system_prompt=ADVANCE_PROMPTS[country], language='en')
    #     save_detailed(output_dir, f"adv_en_{country}", res_adv_en)
    #     s_adv_en = evaluator.aggregate_cultural_scores(res_adv_en, analyzer=analyzer)
    #     summary_data["points"].append({
    #         'RC1': float(s_adv_en['X_Axis']), 'RC2': float(s_adv_en['Y_Axis']), 
    #         'label': f'Adv (EN): {country}', 'color': 'orange'
    #     })
        
    #     # Adv (MLT)
    #     res_adv = evaluator.evaluate_dataset(test_data, system_prompt=ADVANCE_PROMPTS_MLT[country], language=country)
    #     save_detailed(output_dir, f"adv_mlt_{country}", res_adv)
    #     s_adv = evaluator.aggregate_cultural_scores(res_adv, analyzer=analyzer)
    #     summary_data["points"].append({
    #         'RC1': float(s_adv['X_Axis']), 'RC2': float(s_adv['Y_Axis']), 
    #         'label': f'Adv (MLT): {country}', 'color': 'orange'
    #     })

    # --- STEP 3: VECTOR STEERING ---
    print("Training Steering Vectors...")
    vec_x = train_cultural_vector(evaluator.model, train_data, axis='X', batch_size=8)
    vec_y = train_cultural_vector(evaluator.model, train_data, axis='Y', batch_size=8)
    combined_vector = vec_x + vec_y
    release_memory(force=True)
    
    
    # Selection of top layers for subsequent evaluation
    if best_layer_ids is not None:
        best_layers = best_layer_ids
        print(f"Using provided best layers: {best_layers}")
    else:
        # Layer Differential Analysis
        print("Finding best layers for steering...")
        layer_diffs_raw = evaluator.find_best_layers_per_question(combined_vector, train_data)
        summary_data["layer_diffs"] = layer_diffs_raw
        layer_avg = {}
        for q_id, diffs in layer_diffs_raw.items():
            for layer_id, diff in diffs.items():
                layer_avg[layer_id] = layer_avg.get(layer_id, 0) + diff
        best_layers = sorted(layer_avg, key=layer_avg.get, reverse=True)
        print(f"Top layers for steering: {best_layers[:10]}")
        best_layers = sorted(best_layers)[:4]  # Select top 4 layers
    evaluator.model.layer_ids = best_layers

    # Global Vector Steering
    print("Evaluating Global Vector Steering...")
    
    # Steer X and Y separately
    # for coeff in [0.2, -0.2]:
    #     res_vec_x = evaluator.evaluate_dataset(test_data, steering_vector=vec_x, coeff=coeff)
    #     save_detailed(output_dir, f"vector_x_{coeff}", res_vec_x)
    #     scores_vec_x = evaluator.aggregate_cultural_scores(res_vec_x, analyzer=analyzer)
    #     summary_data["vectors"].append({
    #         'RC1': float(scores_vec_x['X_Axis']),
    #         'RC2': float(scores_vec_x['Y_Axis']),
    #         'label': f'Vector Steering (X, {coeff})',
    #         'color': 'blue',
    #         'begin_point_label': 'Baseline'
    #     })
    
    #     res_vec_y = evaluator.evaluate_dataset(test_data, steering_vector=vec_y, coeff=0.2)
    #     save_detailed(output_dir, f"vector_y_{coeff}", res_vec_y)
    #     scores_vec_y = evaluator.aggregate_cultural_scores(res_vec_y, analyzer=analyzer)
    #     summary_data["vectors"].append({
    #         'RC1': float(scores_vec_y['X_Axis']),
    #         'RC2': float(scores_vec_y['Y_Axis']),
    #         'label': f'Vector Steering (Y, {coeff})',
    #         'color': 'blue',
    #         'begin_point_label': 'Baseline'
    #     })

    #     res_steered = evaluator.evaluate_dataset(test_data, steering_vector=combined_vector, coeff=0.2)
    #     save_detailed(output_dir, f"vector_combined_{coeff}", res_steered)
    #     scores_steered = evaluator.aggregate_cultural_scores(res_steered, analyzer=analyzer)
    #     summary_data["vectors"].append({
    #         'RC1': float(scores_steered['X_Axis']), 
    #         'RC2': float(scores_steered['Y_Axis']), 
    #         'label': f'Vector Steering coeff {coeff}', 
    #         'color': 'blue',
    #         'begin_point_label': 'Baseline'
    #     })

    # Combined (Adv MLT + Vector)
    for country in countries_to_use:
        evaluator.model.reset()
        vec_x_basic = train_cultural_vector(evaluator.model, train_data, axis='X', system_prompt=ADVANCE_PROMPTS[country], batch_size=8)
        vec_y_basic = train_cultural_vector(evaluator.model, train_data, axis='Y', system_prompt=ADVANCE_PROMPTS[country], batch_size=8)
        vec_mapping = {
            'vec_x_advance': vec_x_basic,
            'vec_y_advance': vec_y_basic,
            'vec_x+advance': vec_x,
            'vec_y+advance': vec_y,
        }
        for vec_name, vec in vec_mapping.items():
            for coeff in [0.2, -0.2]:
                print(f"[{country}] Combined (Basic + Vector)..."+ f"Vector: {vec_name}, Coeff: {coeff}")
                res_comb = evaluator.evaluate_dataset(test_data, system_prompt=ADVANCE_PROMPTS[country], 
                                                steering_vector=vec, coeff=coeff)
                save_detailed(output_dir, f"vector_{country}_{vec_name}_{coeff}", res_comb)
                s_comb = evaluator.aggregate_cultural_scores(res_comb, analyzer=analyzer)
                summary_data["vectors"].append({
                    'RC1': float(s_comb['X_Axis']), 'RC2': float(s_comb['Y_Axis']), 
                    'label': f'vector_{country}_{vec_name}_{coeff}', 'color': 'green',
                    'begin_point_label': f'Baseline'
                })
                save_summary(output_dir, summary_data)
                del res_comb, s_comb
                release_memory(force=True)

        del vec_x_basic, vec_y_basic, vec_mapping
        release_memory(force=True)
                

    # Domain Shifts (Baseline vs Steered)
    # pivot_baseline = evaluator.get_domain_pivot(res_baseline)
    # pivot_steered = evaluator.get_domain_pivot(res_steered)
    # domain_shifts = (pivot_steered - pivot_baseline).to_dict()
    # summary_data["domain_shifts"] = domain_shifts

    # # --- STEP 4: PERPLEXITY ---
    # print("Measuring Steering Cost...")
    # coeffs = [-0.4, -0.2, 0, 0.2, 0.4]
    # for c in coeffs:
    #     ppl = evaluator.calculate_perplexity(test_data[::5], steering_vector=combined_vector, coeff=c)
    #     summary_data["perplexities"][str(c)] = float(ppl)

    save_summary(output_dir, summary_data)
    release_memory(force=True)
    print(f"All evaluation results saved to {output_dir}")

if __name__ == "__main__":
    # read --model argument from command line
    import argparse
    parser = argparse.ArgumentParser(description="Run the full evaluation pipeline for cultural steering experiments.")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='Model name or path to use for evaluation')
    parser.add_argument('--best-layers', type=str, default=None, help='Comma-separated list of layer IDs to use (e.g., "1,2,3,4"). If not provided, top 4 layers will be automatically selected.')
    parser.add_argument('--test', action='store_true', help='Run in test mode with only the first target country')
    args = parser.parse_args()

    best_layer_ids = None
    if args.best_layers:
        best_layer_ids = [int(x.strip()) for x in args.best_layers.split(',')]
        
    print(f"Using best layers: {best_layer_ids}" if best_layer_ids else "No best layers provided, will select automatically.")
    if args.test:
        print("Running in TEST mode - using only the first target country")
    
    run_paper_experiments(model_name=args.model, best_layer_ids=best_layer_ids, test=args.test)
