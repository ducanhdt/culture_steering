import argparse
import gc
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from src.core.config import ADVANCE_PROMPTS, ADVANCE_PROMPTS_MLT, BASIC_PROMPT_TEMPLATE, DEFAULT_MODEL, DEVICE, TARGET_COUNTRIES
from src.core.evaluator import CulturalEvaluator
from src.core.trainer import train_cultural_vector


def _extract_global_mmlu_choices(example):
    # Try lowercase option_a, option_b, option_c, option_d first (Global MMLU format)
    if all(k in example for k in ["option_a", "option_b", "option_c", "option_d"]):
        choices = [example["option_a"], example["option_b"], example["option_c"], example["option_d"]]
    elif "choices" in example and isinstance(example["choices"], list):
        choices = example["choices"]
    elif "options" in example and isinstance(example["options"], list):
        choices = example["options"]
    elif "choices" in example and isinstance(example["choices"], dict):
        choice_dict = example["choices"]
        ordered_keys = [k for k in ["A", "B", "C", "D"] if k in choice_dict]
        choices = [choice_dict[k] for k in ordered_keys]
    elif all(k in example for k in ["A", "B", "C", "D"]):
        choices = [example["A"], example["B"], example["C"], example["D"]]
    else:
        raise ValueError("Could not infer answer choices from Global MMLU example.")

    if len(choices) < 2:
        raise ValueError("Global MMLU item has fewer than 2 choices.")

    return choices


def _extract_global_mmlu_question(example):
    for key in ["question", "input", "prompt"]:
        if key in example and isinstance(example[key], str) and example[key].strip():
            return example[key].strip()
    raise ValueError("Could not infer question text from Global MMLU example.")


def _extract_global_mmlu_subject(example):
    for key in ["subject", "category", "topic", "domain"]:
        if key in example:
            return str(example[key])
    return "unknown"


def _extract_global_mmlu_language(example):
    for key in ["language", "lang", "locale"]:
        if key in example:
            lang_val = str(example[key]).strip()
            if lang_val and lang_val.lower() != "unknown":
                return lang_val
    # Default to English if no language field
    return "en"


def _extract_global_mmlu_cultural_sensitivity_label(example):
    for key in ["cultural_sensitivity_label", "cultural_label", "sensitivity_label"]:
        if key in example:
            label_val = str(example[key]).strip()
            if label_val:
                return label_val
    # Default if no cultural sensitivity label found
    return "unknown"


def _normalize_global_mmlu_answer(example, choices):
    label_keys = [chr(ord("A") + i) for i in range(len(choices))]
    label_to_idx = {k: i for i, k in enumerate(label_keys)}

    raw_answer = None
    for key in ["answer", "target", "label", "gold"]:
        if key in example:
            raw_answer = example[key]
            break

    if raw_answer is None:
        raise ValueError("Could not infer gold answer from Global MMLU example.")

    if isinstance(raw_answer, (int, np.integer)):
        idx = int(raw_answer)
        if 0 <= idx < len(choices):
            return idx

    if isinstance(raw_answer, str):
        answer_str = raw_answer.strip()

        if answer_str.upper() in label_to_idx:
            return label_to_idx[answer_str.upper()]

        if answer_str.isdigit():
            idx = int(answer_str)
            if 0 <= idx < len(choices):
                return idx

        for idx, choice in enumerate(choices):
            if answer_str == str(choice).strip():
                return idx

    raise ValueError(f"Unsupported gold answer format: {raw_answer}")


def _build_prompt(question, choices, system_prompt=""):
    labels = [chr(ord("A") + i) for i in range(len(choices))]

    lines = []
    if system_prompt and system_prompt.strip():
        lines.append(system_prompt.strip())
        lines.append("")

    lines.append(f"Question: {question}")
    for label, choice in zip(labels, choices):
        lines.append(f"{label}. {choice}")
    lines.append("Answer:")

    return "\n".join(lines)


def _label_token_ids(tokenizer, num_choices):
    ids = []
    for i in range(num_choices):
        label = chr(ord("A") + i)

        # Prefer tokenization with preceding space for decoder-only LMs.
        spaced = tokenizer.encode(" " + label, add_special_tokens=False)
        if spaced:
            ids.append(spaced[-1])
            continue

        plain = tokenizer.encode(label, add_special_tokens=False)
        if not plain:
            raise ValueError(f"Could not encode label token {label}")
        ids.append(plain[-1])

    return ids


def _build_steering_vector(evaluator, steering_mode, steering_train_path, vector_prompt):
    mode = steering_mode.lower()
    if mode == "none":
        return None

    with open(steering_train_path, "r") as f:
        train_data = json.load(f)

    if mode == "x":
        return train_cultural_vector(
            evaluator.model,
            train_data,
            axis="X",
            system_prompt=vector_prompt,
            batch_size=8,
        )

    if mode == "y":
        return train_cultural_vector(
            evaluator.model,
            train_data,
            axis="Y",
            system_prompt=vector_prompt,
            batch_size=8,
        )

    if mode == "xy":
        vec_x = train_cultural_vector(
            evaluator.model,
            train_data,
            axis="X",
            system_prompt=vector_prompt,
            batch_size=8,
        )
        vec_y = train_cultural_vector(
            evaluator.model,
            train_data,
            axis="Y",
            system_prompt=vector_prompt,
            batch_size=8,
        )
        return vec_x + vec_y

    raise ValueError("steering_mode must be one of: none, x, y, xy")


def benchmark_global_mmlu(
    model_name=DEFAULT_MODEL,
    dataset_name="CohereForAI/Global-MMLU",
    max_samples=None,
    output_dir="outputs/global_mmlu",
    system_prompt="",
    steering_mode="none",
    steering_coeff=0.0,
    steering_train_path="data/train_data_mtl.json",
    steering_vector_prompt="",
    languages=None,
    target_country=None,
    best_layers=None,
    config_name="default",
):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Global MMLU benchmark requires the datasets package. Install with: pip install datasets"
        ) from exc

    if system_prompt == "basic":
        system_prompt_full = BASIC_PROMPT_TEMPLATE.format(country=target_country)
    elif system_prompt == "advance":
        system_prompt_full = ADVANCE_PROMPTS[target_country]
    elif system_prompt == "advance_mlt":
        system_prompt_full = ADVANCE_PROMPTS_MLT[target_country]
    else:
        system_prompt_full = system_prompt

    if steering_vector_prompt == "basic":
        vector_prompt_full = BASIC_PROMPT_TEMPLATE.format(country=target_country)
    elif steering_vector_prompt == "advance":
        vector_prompt_full = ADVANCE_PROMPTS[target_country]
    elif steering_vector_prompt == "advance_mlt":
        vector_prompt_full = ADVANCE_PROMPTS_MLT[target_country]
    else:
        vector_prompt_full = steering_vector_prompt

    model_safe_name = model_name.replace("/", "_")
    
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/{model_safe_name}", exist_ok=True)

    evaluator = CulturalEvaluator(model_name)

    print(f"Preparing steering vector for mode {steering_mode}...")
    steering_vector = _build_steering_vector(
        evaluator,
        steering_mode=steering_mode,
        steering_train_path=steering_train_path,
        vector_prompt=vector_prompt_full,
    )
    
    layer_ids = []
    if best_layers:
        if isinstance(best_layers, str):
            layer_ids = [int(x.strip()) for x in best_layers.split(',') if x.strip()]
        elif isinstance(best_layers, list):
            layer_ids = best_layers
            
    evaluator.model.reset()
    if layer_ids:
        evaluator.model.layer_ids = layer_ids
        
    if steering_vector is not None:
        evaluator.model.set_control(steering_vector, steering_coeff)

    # Convert languages to set for faster lookup; default to English
    if languages:
        selected_languages = {lang.strip().lower() for lang in languages if lang and lang.strip()}
    else:
        selected_languages = {"en"}

    def _is_language_selected(lang_value):
        lang_norm = str(lang_value).strip().lower()
        if lang_norm in selected_languages:
            return True
        if ("en" in selected_languages or "english" in selected_languages) and (
            lang_norm.startswith("en") or "english" in lang_norm
        ):
            return True
        return False

    # Try to load the specific language config if only one language is selected
    lang_config = None
    if len(selected_languages) == 1:
        lang_code = list(selected_languages)[0]
        # Common mapping for Global MMLU
        if lang_code in ["en", "vi", "hi", "es", "da", "fr", "de", "zh", "ar", "ru", "pt", "it", "ja", "ko"]:
            lang_config = lang_code

    if lang_config:
        print(f"Loading dataset: {dataset_name} [config={lang_config}, split=test]")
        try:
            dataset = load_dataset(dataset_name, name=lang_config, split="test")
        except Exception as e:
            print(f"Failed to load language config {lang_config}, falling back to default and filtering. Error: {e}")
            dataset = load_dataset(dataset_name, split="test")
            dataset = dataset.filter(lambda example: _is_language_selected(_extract_global_mmlu_language(example)))
    else:
        print(f"Loading dataset: {dataset_name} [split=test]")
        dataset = load_dataset(dataset_name, split="test")
        print(f"Filtering dataset by languages: {selected_languages}")
        dataset = dataset.filter(lambda example: _is_language_selected(_extract_global_mmlu_language(example)))

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    total = 0
    correct = 0
    skipped = 0
    by_subject = {}
    by_language = {}
    by_cultural_sensitivity = {}

    with torch.inference_mode():
        for example in tqdm(dataset, desc="Global MMLU"):
            try:
                question = _extract_global_mmlu_question(example)
                choices = _extract_global_mmlu_choices(example)
                gold_idx = _normalize_global_mmlu_answer(example, choices)
                subject = _extract_global_mmlu_subject(example)
                language = _extract_global_mmlu_language(example)
                cultural_label = _extract_global_mmlu_cultural_sensitivity_label(example)
            except Exception as e:
                skipped += 1
                # print(f"Skipping example due to extraction error: {e}")
                continue

            prompt = _build_prompt(question, choices, system_prompt=system_prompt_full)
            inputs = evaluator.tokenizer(prompt, return_tensors="pt").to(DEVICE)

            outputs = evaluator.model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]

            label_ids = _label_token_ids(evaluator.tokenizer, len(choices))
            choice_logits = next_token_logits[0, label_ids]
            pred_idx = int(torch.argmax(choice_logits).item())

            is_correct = pred_idx == gold_idx
            total += 1
            correct += int(is_correct)

            if subject not in by_subject:
                by_subject[subject] = {"correct": 0, "total": 0}
            by_subject[subject]["correct"] += int(is_correct)
            by_subject[subject]["total"] += 1

            if language not in by_language:
                by_language[language] = {"correct": 0, "total": 0}
            by_language[language]["correct"] += int(is_correct)
            by_language[language]["total"] += 1

            if cultural_label not in by_cultural_sensitivity:
                by_cultural_sensitivity[cultural_label] = {"correct": 0, "total": 0}
            by_cultural_sensitivity[cultural_label]["correct"] += int(is_correct)
            by_cultural_sensitivity[cultural_label]["total"] += 1

            del inputs, outputs, next_token_logits, choice_logits
            if DEVICE == "cuda" and total % 64 == 0:
                torch.cuda.empty_cache()

    accuracy = (correct / total) if total else 0.0
    
    by_subject_accuracy = {
        key: (val["correct"] / val["total"] if val["total"] else 0.0, val["total"])
        for key, val in by_subject.items()
    }
    by_language_accuracy = {
        key: (val["correct"] / val["total"] if val["total"] else 0.0, val["total"])
        for key, val in by_language.items()
    }
    by_cultural_sensitivity_accuracy = {
        key: (val["correct"] / val["total"] if val["total"] else 0.0, val["total"])
        for key, val in by_cultural_sensitivity.items()
    }

    results = {
        "config_name": config_name,
        "model_name": model_name,
        "target_country": target_country,
        "dataset_name": dataset_name,
        "split": "test",
        "num_evaluated": total,
        "num_skipped": skipped,
        "accuracy": accuracy,
        "languages_filter": sorted(selected_languages) if selected_languages else None,
        "system_prompt":    system_prompt_full,
        "steering": {
            "mode": steering_mode,
            "coeff": steering_coeff,
            "train_path": steering_train_path,
            "vector_prompt": vector_prompt_full,
        },
        "by_subject_accuracy": by_subject_accuracy,
        "by_language_accuracy": by_language_accuracy,
        "by_cultural_sensitivity_accuracy": by_cultural_sensitivity_accuracy,
    }

    output_path = os.path.join(
        output_dir, model_safe_name,
        f"mmlu_{config_name}_{target_country}.json",
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Global MMLU complete: {config_name} | {target_country} | Acc: {accuracy:.4f}")

    evaluator.model.reset()
    del evaluator, steering_vector
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Global MMLU benchmark with optional prompt and steering controls")

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--dataset", type=str, default="CohereForAI/Global-MMLU", help="Hugging Face dataset name")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of evaluated samples")
    parser.add_argument("--output-dir", type=str, default="outputs/global_mmlu", help="Output directory")
    parser.add_argument("--best-layers", type=str, default=None, help="Comma-separated list of layer IDs to use")
    parser.add_argument("--test", action="store_true", help="Run in test mode (fewer samples/countries)")

    args = parser.parse_args()

    COUNTRY_TO_LANG = {
        "Denmark": "en",
        "Vietnam": "vi",
        "India": "hi",
        "Mexico": "es"
    }

    configs = [
        {"name": "basic_prompt", "system_prompt": 'basic', "steering_mode": 'none', "steering_coeff": 0.0},
        {"name": "advance_prompt", "system_prompt": 'advance', "steering_mode": 'none', "steering_coeff": 0.0},
        # {"name": "vector_basic_prompt", "system_prompt": 'basic', "steering_mode": 'x', "steering_coeff": 0.2},
        {"name": "vector_advance_prompt", "system_prompt": 'advance', "steering_mode": 'x', "steering_coeff": 0.2},
        {'name': "vector_sp_advance_prompt", "system_prompt": 'advance', "steering_mode": 'x', "steering_coeff": 0.2, "vector_sp": True},
        {'name': "baseline_mlt", "system_prompt": None, "steering_mode": 'none', "steering_coeff": 0.0, "mlt": True},
        {"name": "advance_mlt", "system_prompt": 'advance_mlt', "steering_mode": 'none', "steering_coeff": 0.0, "mlt": True},
        {"name": "vector_advance_mlt", "system_prompt": 'advance_mlt', "steering_mode": 'x', "steering_coeff": 0.2, "mlt": True},
        # {"name": "vector_sp_advance_mlt", "system_prompt": 'advance_mlt', "steering_mode": 'x', "steering_coeff": 0.2, "vector_sp": True, "mlt": True},
    ]

    countries = TARGET_COUNTRIES
    if args.test:
        countries = TARGET_COUNTRIES[:1]
        args.max_samples = args.max_samples or 20

    all_results = []
    default_res = benchmark_global_mmlu(
        model_name=args.model,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        config_name="default")
    all_results.append(default_res)
    for config in configs:
        for country in countries:
            print(f"\n--- Running Config: {config['name']} | Country: {country} ---")
            
            languages = ["en"]
            if config.get("mlt"):
                languages = [COUNTRY_TO_LANG.get(country, "en")]

            vector_prompt = ""
            if config.get("vector_sp"):
                vector_prompt = config.get("system_prompt", "")
            
            res = benchmark_global_mmlu(
                model_name=args.model,
                dataset_name=args.dataset,
                max_samples=args.max_samples,
                output_dir=args.output_dir,
                system_prompt=config.get("system_prompt", ""),
                steering_mode=config.get("steering_mode", "none"),
                steering_coeff=config.get("steering_coeff", 0.0),
                steering_vector_prompt=vector_prompt,
                languages=languages,
                target_country=country,
                best_layers=args.best_layers,
                config_name=config["name"],
            )
            all_results.append(res)

    # Generate Summary CSV
    if all_results:
        import pandas as pd
        summary_rows = []
        for res in all_results:
            row = {
                "config": res["config_name"],
                "country": res["target_country"],
                "total_accuracy": res["accuracy"],
                "num_evaluated": res["num_evaluated"]
            }
            # Flatten by_cultural_sensitivity_accuracy
            for label, (acc, count) in res["by_cultural_sensitivity_accuracy"].items():
                row[f"culture_{label}_acc"] = acc
                row[f"culture_{label}_count"] = count
            summary_rows.append(row)
        model_safe_name = args.model.replace("/", "_")
        df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(args.output_dir, model_safe_name, "global_mmlu_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSummary CSV saved to: {csv_path}")
