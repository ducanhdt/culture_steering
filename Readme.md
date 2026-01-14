# Cultural Values Steering in Large Language Models

This repository contains code for analyzing and steering cultural values in Large Language Models (LLMs) based on the World Values Survey (WVS) and European Values Study (EVS) frameworks, specifically using the Inglehart-Welzel Cultural Map dimensions.

## Project Overview

This research explores how LLMs represent and can be steered toward different cultural value systems across two key dimensions:
- **Traditional vs. Secular-Rational Values**
- **Survival vs. Self-Expression Values**

The project evaluates multiple LLMs (Qwen, Llama, Gemma) across four target countries: India, Vietnam, Mexico, and Denmark.

## Repository Structure

```
├── notebooks/
│   ├── merge.py                      # Merge WVS and EVS datasets
│   ├── generate_data.ipynb           # Generate synthetic scenarios
│   ├── culture-steering.ipynb        # Main steering and evaluation
│   ├── IVS_plot.ipynb               # Visualization of results
│   └── steering_analysis.ipynb       # Analysis of steering effects
├── data/                             # WVS/EVS datasets and generated scenarios
├── outputs/                          # Model outputs and results
├── README.md                         # This file
```

## Setup and Requirements

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (for model inference)

### Python Dependencies

Install required packages:

```bash
pip install torch transformers
pip install pandas numpy matplotlib seaborn
pip install scikit-learn factor-analyzer
pip install dialz  # For steering vectors
pip install pyreadstat  # For SPSS/Stata files
pip install adjustText tqdm
pip install google-generativeai  # For data generation
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
   - Save as: `data/ZA7503_v3-0-0.dta/ZA7503_v3-0-0.dta`

#### 1.2 Merge Datasets

Run the merge script to combine WVS and EVS datasets:

```bash
cd notebooks
python merge.py
```

This script will:
- Load both WVS and EVS trend files
- Standardize column names and types
- Merge datasets by common identifiers
- Handle missing values and dataset-specific variables
- Output: `wvs_evs_trend/ivs_data.pkl`

### Step 2: Generate Synthetic Scenarios

Use the [generate_data.ipynb](notebooks/generate_data.ipynb) notebook to create realistic forced-choice scenarios.

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

### Step 3: Cultural Steering and Evaluation

The [culture-steering.ipynb](notebooks/culture-steering.ipynb) notebook implements the main steering experiments.

#### 3.1 Supported Models

```python
# Choose your model:
model_name = "Qwen/Qwen3-4B-Instruct-2507"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = "google/gemma-3-4b-it"
```

#### 3.2 Steering Methods

**Method 1: Prompt-Based Steering (Basic)**
- Adds country-specific identity in system prompt
- Example: "You are an average person from India"

**Method 2: Prompt-Based Steering (Advanced)**
- Includes detailed cultural descriptions
- Provides context about Traditional/Secular-Rational and Survival/Self-Expression values

**Method 3: Activation Steering**
- Extracts steering vectors from model activations
- Applies vectors at specific layers during inference
- Uses the `dialz` library for implementation

#### 3.3 Evaluation Process

The notebook evaluates models by:
1. Presenting forced-choice scenarios
2. Extracting model probabilities for options A and B
3. Mapping choices to cultural dimensions
4. Comparing with human survey data from target countries

**Key Functions:**
```python
def get_probability(prompt, model, tokenizer):
    # Returns probability distribution over choices A and B
    
def evaluate_scenarios(model, scenarios, country, method):
    # Evaluates model alignment with target country values
```

**Outputs:**
- `qwen_prompt_steer_outputs/` - Prompt steering results
- `qwen_steering_outputs/` - Activation steering results
- Similar folders for Llama and Gemma models

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

#### 4.2 Steering Effect Analysis

Use [steering_analysis.ipynb](notebooks/steering_analysis.ipynb) to analyze activation steering effects.

**What this notebook analyzes:**
- Best layers for steering per WVS dimension
- Magnitude of steering effects across layers
- Differential effects on X-axis vs. Y-axis questions

**Key Outputs:**
- `best_layers_X.csv` - Layer-wise steering effects for Traditional-Secular dimension
- `best_layers_Y.csv` - Layer-wise steering effects for Survival-Self-Expression dimension
- Visualizations of layer effectiveness



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


**Note**: This project requires significant computational resources (GPU with 16GB+ VRAM recommended) and API access (Google Gemini for data generation, Hugging Face for model access).
