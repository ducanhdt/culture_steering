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
    if system_prompt.strip():
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
):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Global MMLU benchmark requires the datasets package. Install with: pip install datasets"
        ) from exc

    if system_prompt in ["basic", "advance"]:
        system_prompt_full = ADVANCE_PROMPTS[country] if args.system_prompt == "advance" else BASIC_PROMPT_TEMPLATE.format(country=country)
    else:
        system_prompt_full = system_prompt

    if steering_vector_prompt in ["basic", "advance"]:
        vector_prompt_full = ADVANCE_PROMPTS_MLT[country] if args.steering_vector_prompt == "advance" else BASIC_PROMPT_TEMPLATE.format(country=country)
    else:
        vector_prompt_full = steering_vector_prompt




    os.makedirs(output_dir, exist_ok=True)

    evaluator = CulturalEvaluator(model_name)

    print("Preparing steering vector...")
    steering_vector = _build_steering_vector(
        evaluator,
        steering_mode=steering_mode,
        steering_train_path=steering_train_path,
        vector_prompt=vector_prompt_full,
    )

    evaluator.model.reset()
    if steering_vector is not None:
        evaluator.model.set_control(steering_vector, steering_coeff)

    # Convert languages to set for faster lookup; default to English
    if languages:
        selected_languages = {lang.strip().lower() for lang in languages if lang and lang.strip()}
        if not selected_languages:
            selected_languages = {"en", "english"}  # Default to English if empty
    else:
        selected_languages = {"en", "english"}  # Default to English when None

    def _is_language_selected(lang_value):
        lang_norm = str(lang_value).strip().lower()
        if lang_norm in selected_languages:
            return True

        # Treat common English variants (e.g., en-US, en_GB) as English.
        if ("en" in selected_languages or "english" in selected_languages) and (
            lang_norm.startswith("en") or "english" in lang_norm
        ):
            return True

        return False

    # Convert selected languages to config name for dataset loading
    # Global MMLU uses language codes like 'en' as configuration
    lang_config = None
    if "en" in selected_languages or "english" in selected_languages:
        lang_config = "en"  # Download English subset directly
    
    if lang_config:
        print(f"Loading dataset: {dataset_name} [config={lang_config}, split=test]")
        dataset = load_dataset(dataset_name, name=lang_config, split="test")
    else:
        # Fallback if language config not available
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
            except Exception:
                skipped += 1
                print(f"Skipping example due to extraction error: {example}")
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
        "model_name": model_name,
        "dataset_name": dataset_name,
        "split": "test",
        "num_evaluated": total,
        "num_skipped": skipped,
        "accuracy": (correct / total) if total else 0.0,
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

    model_safe_name = model_name.replace("/", "_")
    mode_safe = steering_mode.lower()
    output_path = os.path.join(
        output_dir,
        f"global_mmlu_{model_safe_name}_{mode_safe}_{steering_coeff}_{lang_config}_{system_prompt}_{steering_vector_prompt}_{selected_languages}_{target_country}.json",
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Global MMLU benchmark complete. Accuracy: {results['accuracy']:.4f}")
    print(f"Saved results to: {output_path}")

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
    parser.add_argument(
        "--languages",
        type=str,
        default="en",
        help="Comma-separated language subset to evaluate (e.g. 'en,fr,sw').",
    )

    parser.add_argument("--system-prompt", type=str, default="", help="System prompt prepended to every MMLU item")

    parser.add_argument(
        "--steering-mode",
        type=str,
        default="none",
        choices=["none", "x", "y", "xy"],
        help="Steering vector to apply during MMLU evaluation",
    )
    parser.add_argument("--steering-coeff", type=float, default=0.0, help="Steering coefficient")
    parser.add_argument(
        "--steering-train-path",
        type=str,
        default="data/train_data_mtl.json",
        help="Training file used to construct steering vectors",
    )
    parser.add_argument(
        "--steering-vector-prompt",
        type=str,
        default="",
        help="Optional prompt used when constructing steering vectors",
    )

    args = parser.parse_args()

    languages = None
    if args.languages:
        languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]

    if args.system_prompt in ["basic", "advance"] or args.steering_vector_prompt in ["basic", "advance"]:
        for country in TARGET_COUNTRIES:

            benchmark_global_mmlu(
                model_name=args.model,
                dataset_name=args.dataset,
                max_samples=args.max_samples,
                output_dir=args.output_dir,
                system_prompt=args.system_prompt,
                steering_mode=args.steering_mode,
                steering_coeff=args.steering_coeff,
                steering_train_path=args.steering_train_path,
                steering_vector_prompt=args.steering_vector_prompt,
                languages=languages,
                target_country=country
            )
    else:
        benchmark_global_mmlu(
            model_name=args.model,
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            system_prompt=args.system_prompt,
            steering_mode=args.steering_mode,
            steering_coeff=args.steering_coeff,
            steering_train_path=args.steering_train_path,
            steering_vector_prompt=args.steering_vector_prompt,
            languages=languages,
        )
