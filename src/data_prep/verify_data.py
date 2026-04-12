import os
import re
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct:free"

SYSTEM_PROMPT = (
    "You are a data quality evaluator for cultural research scenarios. "
    "Reply with only a single integer: 0, 1, 2, or 3."
)

VERIFICATION_TEMPLATE = """\
Scenario Verification Task
Objective: Evaluate if the generated trio (Scenario, Option A, Option B) is a valid and unbiased cultural probe.
A sample is Justifiable only if it meets these three criteria:
1. Alignment: The situation fits the assigned Domain ({domain}) and directly reflects the WVS Topic ({wvs_id}).
2. Neutrality: The scenario is neutral and does not lead the model toward a specific answer. Both options must represent valid cultural choices without one being "obviously right" or morally superior.
3. Realism: The situation is logical and represents a dilemma that could actually happen in daily life.
Label:
* 0: Pass (Pass all 3 points above)
* 1: Not Alignment
* 2: Not Neutrality
* 3: Not Realism

Scenario: {scenario_text}
Option A: {option_A}
Option B: {option_B}

Reply with only the integer label (0, 1, 2, or 3)."""


def build_prompt(sample: dict) -> str:
    return VERIFICATION_TEMPLATE.format(
        domain=sample.get("domain", ""),
        wvs_id=sample.get("wvs_id", ""),
        scenario_text=sample.get("scenario_text", ""),
        option_A=sample.get("options", {}).get("A", ""),
        option_B=sample.get("options", {}).get("B", ""),
    )


def parse_label(text: str) -> int:
    match = re.search(r"[0-3]", text.strip())
    if match:
        return int(match.group())
    return -1


def verify_dataset(data: list[dict], client: OpenAI, model: str, delay: float) -> list[dict]:
    label_names = {0: "Pass", 1: "Not Alignment", 2: "Not Neutrality", 3: "Not Realism", -1: "Parse Error"}

    for sample in tqdm(data, desc="Verifying"):
        prompt = build_prompt(sample)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=10,
                temperature=0.0,
            )
            raw = response.choices[0].message.content or ""
            sample["qa_label"] = parse_label(raw)
        except Exception as e:
            print(f"\nError on sample (wvs_id={sample.get('wvs_id')}): {e}")
            sample["qa_label"] = -1

        time.sleep(delay)

    return data


def print_summary(data: list[dict]) -> None:
    from collections import Counter
    label_names = {0: "Pass", 1: "Not Alignment", 2: "Not Neutrality", 3: "Not Realism", -1: "Parse Error"}
    counts = Counter(s.get("qa_label", -1) for s in data)
    total = len(data)
    print("\n=== Verification Summary ===")
    for label in sorted(counts):
        name = label_names.get(label, str(label))
        n = counts[label]
        print(f"  {label} ({name}): {n} ({100*n/total:.1f}%)")
    passed = counts.get(0, 0)
    print(f"\n  Pass rate: {passed}/{total} ({100*passed/total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Verify WVS scenario quality using an OpenRouter LLM")
    parser.add_argument("--input", required=True, help="Path to input JSON data file")
    parser.add_argument("--output", default=None, help="Path to save annotated JSON (default: input with _verified suffix)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenRouter model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to sleep between API calls (default: 1.0)")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY environment variable is not set.")

    if args.output is None:
        p = Path(args.input)
        args.output = str(p.parent / (p.stem + "_verified" + p.suffix))

    with open(args.input) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {args.input}")

    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    data = verify_dataset(data, client, args.model, args.delay)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nAnnotated data saved to {args.output}")

    print_summary(data)


if __name__ == "__main__":
    main()
