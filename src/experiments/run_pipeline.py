
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
                         coeffs=[],
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
        'vectors': [],
        'perplexities': {},
        'layer_diffs': {},
        'domain_shifts': {},
        'best_coeffs': {}
    }
    target_means = analyzer.get_target_country_means()
    print(target_means)
    
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
    # save summary
    save_summary(output_dir, summary_data)
    # --- STEP 2: VECTOR STEERING ---
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
        # save best_layers to summary
        summary_data["best_layers"] = best_layers
    evaluator.model.layer_ids = best_layers

    # Global Vector Steering
    print("Evaluating Global Vector Steering...")

    # Process coefficients structure (can be global list or country-specific mapping)
    is_dict_coeffs = isinstance(coeffs, dict)
    do_grid_search = not coeffs
    
    if do_grid_search:
        print("No coefficients provided. Performing grid search for best coefficient per country.")
    else:
        if is_dict_coeffs:
            print(f"Using country-specific coefficients: {coeffs}")
        else:
            print(f"Using provided global coefficients for all countries: {coeffs}")

    # Combined (Adv MLT + Vector)
    for country in countries_to_use:
        evaluator.model.reset()
        # Train vectors
        vec_x_advance = train_cultural_vector(evaluator.model, train_data, axis='X', system_prompt=ADVANCE_PROMPTS[country], batch_size=8, language=country)
        vec_x = train_cultural_vector(evaluator.model, train_data, axis='X', batch_size=8, language=country)
        
        vec_mapping = {
            'vec_x': vec_x,
            'vec_x_advance': vec_x_advance,
        }

        best_coeff_found = {}
        if do_grid_search:
            # Grid search for best coefficient using train_data
            if country in target_means:
                target_rc1, target_rc2 = target_means[country]
                print(f"[{country}] Performing grid search for best coefficient...")
                print("Target coordinate: ", target_rc1, target_rc2)
                
                for vec_name, vec in vec_mapping.items():
                    low, high = 0.0, 0.6
                    print(f"[{country}] Vector {vec_name}: Starting binary (ternary) search in range [{low}, {high}]...")
                    
                    for i in range(4): # 4 iterations
                        c1 = low + (high - low) / 3
                        c2 = high - (high - low) / 3
                        
                        # Evaluate at c1
                        res_search1 = evaluator.evaluate_dataset(train_data, system_prompt=ADVANCE_PROMPTS_MLT[country], 
                                                         steering_vector=vec, coeff=c1, language=country)
                        s_search1 = evaluator.aggregate_cultural_scores(res_search1, analyzer=analyzer)
                        dist1 = np.sqrt((s_search1['X_Axis'] - target_rc1)**2 + (s_search1['Y_Axis'] - target_rc2)**2)
                        
                        # Evaluate at c2
                        res_search2 = evaluator.evaluate_dataset(train_data, system_prompt=ADVANCE_PROMPTS_MLT[country], 
                                                         steering_vector=vec, coeff=c2, language=country)
                        s_search2 = evaluator.aggregate_cultural_scores(res_search2, analyzer=analyzer)
                        dist2 = np.sqrt((s_search2['X_Axis'] - target_rc1)**2 + (s_search2['Y_Axis'] - target_rc2)**2)
                        
                        if dist1 < dist2:
                            high = c2
                            print(f"[{country}] Vector {vec_name}: Iteration {i+1}/4, Range: [{low:.4f}, {high:.4f}] - c1={c1:.4f} (dist: {dist1:.4f}) is better.")
                        else:
                            low = c1
                            print(f"[{country}] Vector {vec_name}: Iteration {i+1}/4, Range: [{low:.4f}, {high:.4f}] - c2={c2:.4f} (dist: {dist2:.4f}) is better.")
                        
                        del res_search1, s_search1, res_search2, s_search2
                        release_memory(force=True)
                    
                    best_c = (low + high) / 2
                    best_coeff_found[vec_name] = best_c
                    print(f"[{country}] Vector {vec_name}: Best coefficient found: {best_c:.4f}")
            else:
                print(f"Warning: No target means found for {country}, using default coefficient 0.2.")
                best_coeff_found = {vec_name: 0.2 for vec_name in vec_mapping.keys()}
        else:
            # Using provided coefficients
            best_coeff_found = {vec_name: None for vec_name in vec_mapping.keys()}

        # Final Evaluation on test_data
        summary_data["best_coeffs"][country] = best_coeff_found
        
        for vec_name, vec in vec_mapping.items():
            if do_grid_search:
                eval_coeffs = [best_coeff_found.get(vec_name, 0.2)]
            elif is_dict_coeffs:
                # Use country specific if available, otherwise fallback to any global ones in the dict (key '*')
                eval_coeffs = coeffs.get(country, coeffs.get('*', []))
                if not eval_coeffs:
                    print(f"Warning: No coefficients specified for {country}, skipping evaluation.")
                    continue
            else:
                eval_coeffs = coeffs
            
            for coeff in eval_coeffs:
                is_best = do_grid_search # if grid search was done, the one being evaluates is "best"
                label_suffix = " (best_grid)" if is_best else ""
                print(f"[{country}] Final Evaluation (Test Data)... Vector: {vec_name}, Coeff: {coeff}{label_suffix}")
                
                res_comb = evaluator.evaluate_dataset(test_data, system_prompt=ADVANCE_PROMPTS_MLT[country], 
                                                steering_vector=vec, coeff=coeff, language=country)
                save_detailed(output_dir, f"vector_{country}_{vec_name}_{coeff}_mlt", res_comb)
                s_comb = evaluator.aggregate_cultural_scores(res_comb, analyzer=analyzer)
                summary_data["vectors"].append({
                    'RC1': float(s_comb['X_Axis']), 'RC2': float(s_comb['Y_Axis']), 
                    'label': f'vector_{country}_{vec_name}_{coeff}{label_suffix}', 'color': 'green' if not is_best else 'darkgreen',
                    'begin_point_label': f'Baseline',
                    'is_best_coeff': is_best
                })
                save_summary(output_dir, summary_data)
                del res_comb, s_comb
                release_memory(force=True)

        del vec_x_advance, vec_x, vec_mapping
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
    parser.add_argument('--coeffs', type=str, default=None, help='Comma-separated list of steering coefficients (e.g., "0.2,-0.2"). If not provided, grid search will be performed.')
    parser.add_argument('--test', action='store_true', help='Run in test mode with only the first target country')
    args = parser.parse_args()

    best_layer_ids = None
    if args.best_layers:
        best_layer_ids = [int(x.strip()) for x in args.best_layers.split(',')]
        
    coeffs = [] # Default to grid search
    if args.coeffs:
        if ':' in args.coeffs:
            # Handle country-specific mapping: Vietnam:0.1,0.2;Denmark:0.3
            coeffs = {}
            parts = args.coeffs.split(';')
            for part in parts:
                if ':' in part:
                    country, vals = part.split(':')
                    coeffs[country.strip()] = [float(v.strip()) for v in vals.split(',')]
                else:
                    # Global values without a prefix
                    coeffs['*'] = [float(v.strip()) for v in part.split(',')]
        else:
            # Standard global list
            coeffs = [float(x.strip()) for x in args.coeffs.split(',')]
        
    print(f"Using best layers: {best_layer_ids}" if best_layer_ids else "No best layers provided, will select automatically.")
    print(f"Using steering coefficients: {coeffs}")
    if args.test:
        print("Running in TEST mode - using only the first target country")
    
    run_paper_experiments(model_name=args.model, best_layer_ids=best_layer_ids, coeffs=coeffs, test=args.test)
