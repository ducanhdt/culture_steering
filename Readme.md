# Cultural Values Steering in Large Language Models

This repository contains code for analyzing and steering cultural values in Large Language Models (LLMs) based on the World Values Survey (WVS) and European Values Study (EVS) frameworks, specifically using the Inglehart-Welzel Cultural Map dimensions.

## Project Overview

This research explores how LLMs represent and can be steered toward different cultural value systems across two key dimensions:
- **Traditional vs. Secular-Rational Values**
- **Survival vs. Self-Expression Values**

The project evaluates multiple LLMs (Qwen, Llama, Gemma) across four target countries: India, Vietnam, Mexico, and Denmark.

## Repository Structure

```
├── src/
│   ├── data_prep/
│   │   ├── merge.py                          # Merge WVS + EVS into ivs_data_processed.pkl
│   │   └── generate_data.py                  # Headless scenario generation via Gemini
│   ├── core/
│   │   ├── config.py                         # WVS IDs, target countries, prompt templates
│   │   ├── trainer.py                        # train_cultural_vector (dialz mean-diff)
│   │   └── evaluator.py                      # CulturalEvaluator + layer differential analysis
│   ├── utils/data_utils.py                   # WVSAnalyzer (varimax FA → RC1/RC2 projection)
│   ├── experiments/
│   │   ├── run_pipeline_hybrid_search.py     # Layer selection + per-country coeff search
│   │   ├── run_fixed_pipeline.py             # All configs × all countries at fixed coeffs
│   │   ├── run_pipeline_vector_grid_search.py# Coefficient grid sweep
│   │   ├── run_global_mmlu.py                # Global MMLU accuracy under each config
│   │   └── run_significance.py               # Bootstrap + Holm significance table
│   └── analysis/
│       ├── significance.py                   # Bootstrap + Holm primitives
│       ├── plotting.py
│       └── paper_plots.py
├── notebooks/
│   ├── generate_data.ipynb                   # Interactive scenario generation
│   ├── culture-steering.ipynb                # Exploratory steering / evaluation
│   ├── IVS_plot.ipynb                        # Cultural-map visualization
│   ├── IVS_significance_ellipses.ipynb       # 95% bootstrap ellipses on the cultural map
│   ├── steering_analysis.ipynb               # Best-layer & magnitude analysis
│   └── domain_analysis.ipynb                 # Per-domain shifts & entanglement
├── data/                                     # WVS/EVS source + generated scenarios
├── wvs_evs_trend/                            # WVS trend file + merged ivs_data_processed.pkl
├── outputs/                                  # summary_results*.json + details/ per model
└── Readme.md
```

## Setup and Requirements

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (for model inference)

### Python Dependencies

Install required packages:

```bash
pip install -r requirements.txt
pip install git+https://github.com/ducanhdt/dialz   # steering library (fork)
```

Or install piecewise:

```bash
pip install torch transformers
pip install pandas numpy matplotlib seaborn
pip install scikit-learn factor-analyzer
pip install pyreadstat            # SPSS/Stata files
pip install adjustText tqdm
pip install google-generativeai   # data generation (Gemini)
pip install git+https://github.com/ducanhdt/dialz
```

For notebooks running on Kaggle/Colab:
```bash
pip install -U protobuf==5.29.3
```

### Hugging Face Authentication

You'll need a Hugging Face token to access gated models:

```bash
huggingface-cli login --token YOUR_HF_TOKEN
```

## Workflow

### Step 1: Download and Prepare Data

#### 1.1 Download WVS and EVS Data

Download the following datasets:

1. **WVS Trend File (1981-2022)**
   - Download from: [World Values Survey](https://www.worldvaluessurvey.org/)
   - Save as: `wvs_evs_trend/Trends_VS_1981_2022_sav_v4_0.sav`

2. **EVS Data (Version 3.0.0)**
   - Download from: [European Values Study](https://europeanvaluesstudy.eu/)
   - Save as: `wvs_evs_trend/ZA7503_v3-0-0.dta/ZA7503_v3-0-0.dta`

#### 1.2 Merge Datasets

Run the merge script to combine WVS and EVS datasets:

```bash
PYTHONPATH=. python src/data_prep/merge.py
```

This script will:
- Load both WVS and EVS trend files
- Standardize column names and types
- Merge datasets by common identifiers
- Handle missing values and dataset-specific variables
- Extract relevant variables for the Inglehart-Welzel dimensions
- Output: `wvs_evs_trend/ivs_data_processed.pkl`

### Step 2: Generate Synthetic Scenarios

Use the [generate_data.ipynb](notebooks/generate_data.ipynb) notebook to create realistic forced-choice scenarios. For headless / batch runs, the same logic is available as a script at [src/data_prep/generate_data.py](src/data_prep/generate_data.py).

**What this notebook does:**
- Uses Google's Gemini API to generate scenarios based on WVS dimensions
- Creates scenarios across three domains: Workplace, Family, and Legal
- Covers all 10 WVS questions (5 per dimension)
- Generates training and test datasets

**Configuration:**
1. Add your Google API key:
   ```python
   client = genai.Client(api_key="YOUR_API_KEY_HERE")
   ```

2. Adjust generation parameters:
   ```python
   test_scenarios = generate_scenarios(repeats=5)
   train_scenarios = generate_scenarios(repeats=5)
   ```

**Outputs:**
- `data/generated_wvs_scenarios_auto_train.json` - Training scenarios
- `data/generated_wvs_scenarios_auto_test.json` - Test scenarios

### Step 3: Cultural Steering and Evaluation (`src/experiments/`)

The production pipeline lives in `src/experiments/`. All runners take `--model`, accept `--best-layers "20,21,22,23"` (comma-separated layer IDs), and support `--test` for a quick smoke run. Results are written under `outputs/<model_safe_name>/` (`<model_safe_name>` is the HF id with `/` → `_`), with per-condition detail JSONs in `outputs/<model_safe_name>/details/`.

#### 3.1 Supported Models

```bash
--model Qwen/Qwen3-4B-Instruct-2507
--model meta-llama/Llama-3.2-3B-Instruct
--model google/gemma-3-4b-it
```

#### 3.2 Steering Methods

- **Default** — no steering, no country prompt (baseline).
- **Basic prompt** — adds country identity, e.g. *"You are an average person from India."*
- **Advanced prompt** — adds detailed cultural description across Traditional/Secular-Rational and Survival/Self-Expression.
- **Activation steering (`vector`)** — `dialz`-trained mean-difference steering vectors applied at the selected layers; coefficient controls the direction/magnitude per axis.
- **`vector + advanced`** — activation steering on top of the advanced prompt.

#### 3.3 Runners

**Find best layers and per-country coefficients** ([run_pipeline_hybrid_search.py](src/experiments/run_pipeline_hybrid_search.py)). Trains the X/Y vectors, runs layer differential analysis to pick the top-4 layers (or honors `--best-layers`), then does binary/ternary search per country for the best coefficient. Writes `outputs/<model>/summary_results.json`.

```bash
PYTHONPATH=. python src/experiments/run_pipeline_hybrid_search.py \
    --model Qwen/Qwen3-0.6B --best-layers "20,21,22,23"
```

**Run all configurations at fixed coefficients** ([run_fixed_pipeline.py](src/experiments/run_fixed_pipeline.py)). Evaluates baseline / basic prompt / advanced prompt / vector / vector+advanced across all four countries. Writes `summary_results_fixed_pipeline.json` plus per-config detail JSONs.

```bash
PYTHONPATH=. python src/experiments/run_fixed_pipeline.py \
    --model Qwen/Qwen3-0.6B --best-layers "20,21,22,23" --coeffs "0.2,-0.2"
```

**Sweep the steering coefficient grid** ([run_pipeline_vector_grid_search.py](src/experiments/run_pipeline_vector_grid_search.py)) — populates `outputs_grid_coeff/<model>/`. Writes `summary_results_vector_grid_search.json`.

```bash
PYTHONPATH=. python src/experiments/run_pipeline_vector_grid_search.py \
    --model Qwen/Qwen3-0.6B --best-layers "20,21,22,23" --coeffs "0.1,0.2,0.3"
```

**Global MMLU benchmark under each steering config** ([run_global_mmlu.py](src/experiments/run_global_mmlu.py)).

```bash
PYTHONPATH=. python src/experiments/run_global_mmlu.py \
    --model Qwen/Qwen3-0.6B --max-samples 100 --best-layers "20,21,22,23"
```

**Bootstrap + Holm significance table** ([run_significance.py](src/experiments/run_significance.py)) — see Section 4.2.

```bash
PYTHONPATH=. python src/experiments/run_significance.py
```

### Step 4: Visualization and Analysis

#### 4.1 Inglehart-Welzel Cultural Map Plotting

Use [IVS_plot.ipynb](notebooks/IVS_plot.ipynb) to visualize model positions on the cultural map.

**What this notebook does:**
- Loads evaluation results for all models and steering methods
- Performs factor analysis to project onto X (Traditional-Secular) and Y (Survival-Self-Expression) axes
- Plots model positions against human baseline data
- Compares steering effectiveness across methods

**Key Visualizations:**
- Cultural map with model positions
- Comparison of steering methods
- Distance metrics from target country values

**Analysis includes:**
- Direct profile responses (answering WVS questions directly)
- Probing profile responses (forced-choice scenarios)
- Steering effectiveness across different countries

#### 4.2 Statistical Significance (Bootstrap + Holm)

Use [src/analysis/significance.py](src/analysis/significance.py) and the runner [src/experiments/run_significance.py](src/experiments/run_significance.py) to test whether each steering method moves the model significantly closer to its target country on the cultural map. The companion notebook [notebooks/IVS_significance_ellipses.ipynb](notebooks/IVS_significance_ellipses.ipynb) renders 95% confidence ellipses around each condition.

**Method.** For each (model, country, method) detail JSON (300 forced-choice items), we resample the 300 rows with replacement `N_boot = 10,000` times. For each resample we recompute the 10 per-question means → project to (RC1, RC2) via the same fitted varimax FA used in `WVSAnalyzer` → get a Euclidean distance to the country's human-baseline (RC1, RC2). Where the baseline and steered files share item ordering we use a **paired** bootstrap (same indices applied to both); otherwise it falls back to unpaired and reports `pairing` in the output.

**Test.** One-sided null: steering does not reduce distance. `p = (1 + #{Δ ≤ 0}) / (N_boot + 1)`, where `Δ_b = dist_baseline_b − dist_method_b`. The 95% CI on Δ is the 2.5/97.5 percentile.

**Multiple comparisons.** Within each country, the ~12 raw p-values across (model × method) are corrected with **Holm**. A method is declared significant for that country iff `p_holm < 0.05`.

Run:

```bash
PYTHONPATH=. python src/experiments/run_significance.py
```

**Output: [outputs/significance_table.csv](outputs/significance_table.csv)**

| Column | Meaning |
|---|---|
| `model`, `country`, `method` | The condition under test (`method = baseline` is the reference and is dropped from the table; methods are `basic_prompt`, `advanced_prompt`, `vector`, `vector+advanced`). |
| `pairing` | `paired` if baseline and method JSONs share item ordering, else `unpaired`. Paired tests have more power. |
| `dist_method_mean` | Bootstrap-mean Euclidean distance from the method's (RC1, RC2) to the target country's human (RC1, RC2). Smaller = method lands closer to the target. |
| `dist_baseline_mean` | Same, for the model's default profile (no steering). |
| `delta_mean` | `dist_baseline_mean − dist_method_mean` averaged over bootstrap iterations. **Positive = steering pulls the model toward the target.** |
| `delta_ci_low`, `delta_ci_high` | 95% bootstrap percentile CI on Δ. CI excluding 0 ⇔ significant before correction. |
| `p_raw` | One-sided bootstrap p-value for H₀: Δ ≤ 0. |
| `p_holm` | Holm-adjusted p-value within the country's family (12 tests). |
| `significant` | `p_holm < 0.05`. |

**How to read a row.** Look at `delta_mean` for direction and effect size, `delta_ci_low`/`delta_ci_high` for uncertainty, and `p_holm` for the family-corrected significance verdict. Example: `Qwen3-4B-Instruct, Vietnam, vector+advanced`, `delta_mean = 2.76`, CI `[2.41, 3.08]`, `p_holm ≈ 0.001` → vector+advanced steering closes ≈ 2.76 RC-units of distance from baseline to Vietnam, robustly.

**Headline findings.**
- **Activation steering (`vector`) is significant in every country for every model** (12/12 cells).
- `vector+advanced` is significant in 11/12 cells (only Gemma–Denmark fails).
- Prompt-only methods are mixed: `advanced_prompt` is significant for Gemma in all four countries and for Qwen on India; `basic_prompt` is rarely significant.

**95% confidence ellipses.** Run [notebooks/IVS_significance_ellipses.ipynb](notebooks/IVS_significance_ellipses.ipynb). It bootstraps each condition's (RC1, RC2) at `N_boot = 2000`, fits a 2-D Gaussian to the bootstrap cloud, and draws the chi²(2)=5.991 ellipse on the cultural map (one panel per target country). Saved as `outputs/significance_ellipses.png`.

#### 4.3 Steering Effect Analysis

Use [steering_analysis.ipynb](notebooks/steering_analysis.ipynb) to analyze activation steering effects.

**What this notebook analyzes:**
- Best layers for steering per WVS dimension
- Magnitude of steering effects across layers
- Differential effects on X-axis vs. Y-axis questions

**Key Outputs:**
- `best_layers_X.csv` - Layer-wise steering effects for Traditional-Secular dimension
- `best_layers_Y.csv` - Layer-wise steering effects for Survival-Self-Expression dimension
- Visualizations of layer effectiveness

#### 4.4 Per-Domain Steering Analysis

Use [notebooks/domain_analysis.ipynb](notebooks/domain_analysis.ipynb) to inspect how steering effects vary across the three scenario domains (**Family**, **Legal**, **Workplace**) and compare against the combined (**All**) condition. The notebook is the cleaned-up domain slice of [IVS_plot.ipynb](notebooks/IVS_plot.ipynb).

**What this notebook does:**
- Loads default (unsteered) probing profiles for Qwen, Llama, and Gemma and assembles them into `probing_points_df`.
- Defines `compare_profile_on_axes(...)` and `add_entanglement_ratios(...)` helpers that compute, per domain, the **X-shift** (mean change on Traditional↔Secular questions), the **Y-shift** (Survival↔Self-Expression), the **Entanglement Ratio** `min(|ΔX|,|ΔY|)/max(|ΔX|,|ΔY|)` (1 = fully entangled, 0 = clean single-axis steering), and the total Euclidean magnitude.
- Plots per-model and combined `domain × method` heatmaps for fixed-coefficient X = ±0.2 steering — bottom of the notebook contains a 2×3 grid (rows = X/Y axis, columns = Qwen/Llama/Gemma) with shared color scale.
- Aggregates entanglement ratios over the full coefficient sweep in `outputs_grid_coeff/` (positive coefficients only) and over the **best-coeff binary-search** results in `outputs_binary_search/`, reporting `mean ± std` per `(model, domain)`.
- Final section (`Domain-analysis-specific for best-coeff binary-search results`) builds **`domain × target country` heatmaps** for each model using only the per-country best coefficient. Shifts are normalized by the per-axis gap from the model's base point to the human-baseline (RC1, RC2) of the target country (from `pca_result_country_level`), so a cell value of `1.0` means the steering closed that axis's gap exactly, `0` is no movement, and >1 is overshoot.

**Inputs (already produced by earlier steps):**
- `qwen_prompt_steer_outputs_4/`, `llama_prompt_steer_outputs/`, `gemma_prompt_steer_outputs/` — default profiles per model
- `qwen_steering_outputs_4/`, `llama_steering_outputs/`, `gemma_steering_outputs/` — per-domain X = ±0.2 results
- `outputs_grid_coeff/<model>/` — coefficient sweep `summary_results.json` + `details/`
- `outputs_binary_search/<model>/` — best-coeff per country `summary_results.json` + `details/`

**How to read it:** low entanglement on Legal/Workplace means the steering vector moves the targeted axis without dragging the orthogonal axis along; high entanglement (close to 1) means the two axes shift together and the vector is not axis-pure. The final normalized heatmap shows how much of the *country-specific* gap on each axis the best-coeff steering actually closes, broken down by domain.


### Question Mappings

| ID | Dimension | Concept |
|----|-----------|---------|
| F063 | X | Importance of God |
| Y003 | X | Autonomy Index (Child values) |
| F120 | X | Justifiability of Abortion |
| G006 | X | National Pride |
| E018 | X | Respect for Authority |
| A008 | Y | Happiness |
| A165 | Y | Trust in People |
| E025 | Y | Petition Signing |
| F118 | Y | Justifiability of Homosexuality |
| Y002 | Y | Security vs. Expression Priority |


## Troubleshooting

### Common Issues

**Issue 1: Memory errors during model loading**
- Solution: Use smaller models or enable CPU offloading
- Add: `device_map="auto"` in model loading

**Issue 2: API rate limits (Google Gemini)**
- Solution: Increase sleep time between batches
- Adjust: `time.sleep(60)` in generation loop

**Issue 3: Missing data files**
- Ensure all WVS/EVS files are downloaded to correct paths
- Check `.gitignore` - CSV and JSON files are excluded from git

**Issue 4: Hugging Face authentication**
- Get token from: https://huggingface.co/settings/tokens
- Run: `huggingface-cli login --token YOUR_TOKEN`

