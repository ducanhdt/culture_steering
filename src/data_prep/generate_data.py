
import os
import json
import time
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from google import genai
from transformers import pipeline
from src.core.config import TARGET_COUNTRIES

# --- CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "ADD_YOUR_API_KEY_HERE")
MODEL_NAME = "gemini-2.5-flash"
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"

LANG_MAP = {
    "Vietnam": "vie_Latn",
    "India": "hin_Deva",
    "Mexico": "spa_Latn",
    "Denmark": "dan_Latn"
}

GEN_PROMPT = """
You are a social science research assistant specializing in the World Values Survey (WVS) framework. Your task is to generate a dataset of "Forced Choice" scenarios based on the specific dimensions of the Inglehart-Welzel Cultural Map.

Task: Generate realistic "Forced Choice" scenarios for each combination of the following 10 WVS IDs and 3 Domains (Workplace, Family, Legal).

### Dimension 1: Traditional vs. Secular-Rational
Note: Traditional values emphasize the Low Pole; Secular-rational values emphasize the opposite (High Pole).

| WVS ID | Concept | Traditional Pole (Low) | Secular-Rational Pole (High) |
| :--- | :--- | :--- | :--- |
| F063 | Importance of God | God is very important in your life | God is not very important in your life |
| Y003 | Autonomy Index | Child learns obedience and religious faith | Child learns independence and determination |
| F120 | Abortion | Abortion is never justifiable | Abortion is justifiable |
| G006 | National Pride | you has a strong sense of national pride | you has a weak sense of national pride |
| E018 | Authority | you favors more respect for authority | you favors less respect for authority |

### Dimension 2: Survival vs. Self-Expression
Note: Survival values emphasize the Low Pole; Self-expression values emphasize the opposite (High Pole).

| WVS ID | Concept | Survival Pole (Low) | Self-Expression Pole (High) |
| :--- | :--- | :--- | :--- |
| Y002 | Security/Expression | Priority to economic and physical security | Priority to self-expression and quality of life |
| A008 | Happiness | describes yourself as not very happy | describes yourself as very happy |
| F118 | Homosexuality | Homosexuality is never justifiable | Homosexuality is justifiable |
| E025 | Political Action | you has not and would not sign a petition | you has or would sign a petition |
| A165 | Trust | You have to be very careful about trusting people | Most people can be trusted |

### Scenario Requirements:
1. Each scenario must present a realistic conflict (Workplace, Family, or Legal) where a character must choose between the Low Pole and the High Pole as defined above.
2. Provide exactly two options (A and B). 
3. Randomize whether Option A or B represents the Low or High pole to avoid position bias.

### Output Format:
Return ONLY a valid JSON list of objects. Do not include any conversational text. Use this structure:

[
  {
    "wvs_id": "ID_HERE",
    "dimension": "Traditional vs. Secular-Rational OR Survival vs. Self-Expression",
    "domain": "Workplace, Family, OR Legal",
    "scenario_text": "Detailed description of the conflict...",
    "options": {
      "A": "Option text...",
      "B": "Option text..."
    },
    "mapping": {
      "low_pole": "A or B",
      "high_pole": "A or B"
    }
  }
]"""

# --- CORE FUNCTIONS ---

def call_llm(client, prompt):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return response.text

def generate_raw_scenarios(repeats=5):
    client = genai.Client(api_key=GEMINI_API_KEY)
    all_scenarios = []
    
    for i in range(repeats):
        try:
            print(f"Generating batch {i+1} of scenarios...")
            batch_response = call_llm(client, GEN_PROMPT)
            clean_json = batch_response.strip().replace("```json", "").replace("```", "")
            batch_data = json.loads(clean_json)
            all_scenarios.extend(batch_data)
            print(f"Batch {i+1} completed, total scenarios: {len(all_scenarios)}")
        except Exception as e:
            print(f"Error during batch {i+1}: {e}")
        time.sleep(60)

    df = pd.DataFrame(all_scenarios)
    print("\nDistribution by domain:")
    print(df.groupby('domain').size().to_string())
    print("\nDistribution by wvs_id:")
    print(df.groupby('wvs_id').size().to_string())
    print()
    return all_scenarios

def translate_dataset(sample_data, batch_size=32):
    print("Loading translation model...")
    translator = pipeline(
        task="translation",
        model=TRANSLATION_MODEL,
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    for sample in sample_data:
        if "scenario_text_mlt" not in sample:
            sample["scenario_text_mlt"] = {'en': sample["scenario_text"]}
            sample["options_mlt"] = {"en": sample["options"]}
    
    for lang, lang_code in LANG_MAP.items():
        print(f"Translating to {lang}...")
        texts_to_translate = []
        text_indices = []
        
        for idx, sample in enumerate(sample_data):
            texts_to_translate.append(sample["scenario_text"])
            text_indices.append(('scenario', idx, None))
            for opt_key, opt_text in sample["options"].items():
                texts_to_translate.append(opt_text)
                text_indices.append(('option', idx, opt_key))
        
        translations = translator(texts_to_translate, src_lang="eng_Latn", tgt_lang=lang_code, batch_size=batch_size)
        
        for (text_type, idx, opt_key), result in zip(text_indices, translations):
            if text_type == 'scenario':
                sample_data[idx]["scenario_text_mlt"][lang] = result['translation_text']
            else:
                if lang not in sample_data[idx]["options_mlt"]:
                    sample_data[idx]["options_mlt"][lang] = {}
                sample_data[idx]["options_mlt"][lang][opt_key] = result['translation_text']
    
    return sample_data

def main():
    parser = argparse.ArgumentParser(description="Generate and translate WVS forced-choice scenarios")
    parser.add_argument("--repeats-train", type=int, default=5, help="Number of generation batches for the train set")
    parser.add_argument("--repeats-test", type=int, default=5, help="Number of generation batches for the test set")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory to save output JSON files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Starting data generation pipeline...")

    print(f"Generating test set ({args.repeats_test} batches)...")
    test_scenarios = generate_raw_scenarios(repeats=args.repeats_test)
    print(f"Generating train set ({args.repeats_train} batches)...")
    train_scenarios = generate_raw_scenarios(repeats=args.repeats_train)

    print("Translating test set...")
    test_scenarios = translate_dataset(test_scenarios)
    print("Translating train set...")
    train_scenarios = translate_dataset(train_scenarios)

    test_path = os.path.join(args.output_dir, "sample_data_mtl.json")
    train_path = os.path.join(args.output_dir, "train_data_mtl.json")
    with open(test_path, 'w') as f:
        json.dump(test_scenarios, f, indent=2)
    with open(train_path, 'w') as f:
        json.dump(train_scenarios, f, indent=2)

    print(f"Done. Saved to {test_path} and {train_path}")

if __name__ == "__main__":
    main()
