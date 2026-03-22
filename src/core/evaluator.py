
import json
import torch
import numpy as np
import pandas as pd
import copy
import gc
from tqdm import tqdm
from transformers import AutoTokenizer
from dialz import SteeringModel, SteeringVector
from src.core.config import DEVICE, X_AXIS_ID, Y_AXIS_ID, QUESTION_ENDINGS, get_num_layers

class CulturalEvaluator:
    def __init__(self, model_name, layer_ids=None, id_to_info=None):
        self.model_name = model_name
        num_layers = get_num_layers(model_name)
        self.layer_ids = layer_ids or list(range(1, num_layers))
        self.model = SteeringModel(model_name, layer_ids=self.layer_ids)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(DEVICE)
        self.id_to_info = id_to_info

    def _cleanup_memory(self, force=False):
        """Release Python refs and ask CUDA allocator to drop cached blocks."""
        gc.collect()
        if DEVICE == "cuda" and force:
            torch.cuda.empty_cache()

    def evaluate_dataset(self, dataset, system_prompt="", steering_vector=None, coeff=0.1, language=None, batch_size=1):
        """
        Runs batch evaluation on a dataset.
        """
        self.model.reset()
        if steering_vector:
            self.model.set_control(steering_vector, coeff)

        ending = QUESTION_ENDINGS.get(language, QUESTION_ENDINGS['en'])
        prompts = []
        for entry in dataset:
            if language is None or language == 'en':
                scenario_text = entry['scenario_text']
                option_A = entry['options']['A']
                option_B = entry['options']['B']
            else:
                scenario_text = entry['scenario_text_mlt'][language]
                option_A = entry['options_mlt'][language]['A']
                option_B = entry['options_mlt'][language]['B']

            prompt = f"{system_prompt}\n{scenario_text}\nA) {option_A}\nB) {option_B}\n\n{ending}"
            prompts.append(prompt)

        # Batch Tokenization
        token_id_A = self.tokenizer.encode("A", add_special_tokens=False)[-1]
        token_id_B = self.tokenizer.encode("B", add_special_tokens=False)[-1]
        
        all_probs_A = []
        all_probs_B = []

        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i + batch_size]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(DEVICE)
            
            with torch.inference_mode():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :] # [Batch, Vocab]
                
                choice_logits = logits[:, [token_id_A, token_id_B]]
                probs = torch.nn.functional.softmax(choice_logits, dim=-1)
                probs_cpu = probs.detach().cpu()
                all_probs_A.extend(probs_cpu[:, 0].tolist())
                all_probs_B.extend(probs_cpu[:, 1].tolist())

            del inputs, outputs, logits, choice_logits, probs, probs_cpu
            if DEVICE == "cuda" and ((i // batch_size + 1) % 32 == 0):
                self._cleanup_memory(force=True)

        results = copy.deepcopy(dataset)
        for idx, res in enumerate(results):
            prob_A, prob_B = all_probs_A[idx], all_probs_B[idx]
            
            # Mapping logic
            choose_low_pole = prob_A > prob_B if res['mapping']['low_pole'] == "A" else prob_B > prob_A
            wvs_info = self.id_to_info[res['wvs_id']]
            is_high_pole_at_max = wvs_info['direction']
            scale_min, scale_max = wvs_info['Scale_Min'], wvs_info['Scale_Max']

            prob_high_pole = prob_B if res['mapping']['low_pole'] == "A" else prob_A
            prob_low_pole = prob_A if res['mapping']['low_pole'] == "A" else prob_B

            if is_high_pole_at_max:
                score = ((1 - prob_high_pole) * scale_min) + (prob_high_pole * scale_max)
            else:
                score = ((1 - prob_low_pole) * scale_max) + (prob_low_pole * scale_min)

            res.update({
                "prob_high_pole": prob_high_pole, "prob_low_pole": prob_low_pole,
                "human_aligned_score": score, "normalized_score": (score - scale_min) / (scale_max - scale_min)
            })
            res["choice"] = "Traditional" if choose_low_pole else "Secular-Rational" if res['dimension'] == "Traditional vs. Secular-Rational Values" else "Survival" if choose_low_pole else "Self-Expression"
            
        self.model.reset()
        self._cleanup_memory(force=True)
        return results

    def get_domain_pivot(self, results, value_col='normalized_score'):
        """
        Pivots results into a DataFrame with domains as index and WVS IDs as columns.
        """
        df = pd.DataFrame(results)
        pivot = df.pivot_table(index='domain', columns='wvs_id', values=value_col, aggfunc='mean')
        return pivot

    def project_onto_cultural_map(self, results, analyzer):
        """
        Uses a WVSAnalyzer to project raw scores onto RC1/RC2 axes.
        """
        # Get mean score for each WVS ID across all results
        df = pd.DataFrame(results)
        means = df.groupby('wvs_id')['human_aligned_score'].mean().to_frame().T
        # analyzer expects columns in specific order
        rc1, rc2 = analyzer.project_scores(means)
        return rc1[0], rc2[0]

    def aggregate_cultural_scores(self, results, analyzer=None):
        """Converts raw results into cultural axis scores."""
        if analyzer is not None:
            rc1, rc2 = self.project_onto_cultural_map(results, analyzer)
            return {"X_Axis": rc1, "Y_Axis": rc2}
            
        x_scores, y_scores = [], []
        for res in results:
            score = res['normalized_score']
            if res["wvs_id"] in X_AXIS_ID:
                x_scores.append(score)
            elif res["wvs_id"] in Y_AXIS_ID:
                y_scores.append(score)
                
        return {
            "X_Axis": np.mean(x_scores) if x_scores else 0,
            "Y_Axis": np.mean(y_scores) if y_scores else 0
        }

    def find_best_layers_per_question(self, steering_vector, dataset, coeff=0.1):
        """
        Calculates the differential effect of each layer for each question in the dataset.
        Returns a dictionary of {question_id: {layer_id: differential}}
        """
        # evaluate_dataset handles reset internally; no steering = clean baseline
        baseline_results = self.evaluate_dataset(dataset)
        baseline_scores = {res['wvs_id']: res['normalized_score'] for res in baseline_results}

        layer_differentials = {}
        for layer_id in tqdm(self.layer_ids, desc="Finding Best Layers"):
            # Set single layer BEFORE calling evaluate_dataset.
            # evaluate_dataset resets then re-applies the steering vector using
            # the current model.layer_ids — so the order here matters.
            self.model.layer_ids = [layer_id]
            steered_results = self.evaluate_dataset(dataset, steering_vector=steering_vector, coeff=coeff)

            for res in steered_results:
                q_id = res['wvs_id']
                diff = abs(res['normalized_score'] - baseline_scores[q_id])
                if q_id not in layer_differentials:
                    layer_differentials[q_id] = {}
                layer_differentials[q_id][layer_id] = diff

            self._cleanup_memory(force=DEVICE == "cuda")

        # Restore full layer list
        self.model.layer_ids = self.layer_ids
        self.model.reset()
        self._cleanup_memory(force=True)
        return layer_differentials

    def calculate_perplexity(self, dataset, system_prompt="", steering_vector=None, coeff=0.1, language=None):
        """
        Calculates the perplexity of the model's responses to evaluate steering 'cost'.
        """
        self.model.reset()
        if steering_vector:
            self.model.set_control(steering_vector, coeff)

        perplexities = []
        ending = QUESTION_ENDINGS.get(language, QUESTION_ENDINGS['en'])
        
        for entry in tqdm(dataset, desc=f"Perplexity ({language if language else 'en'})"):
            if language is None or language == 'en':
                scenario_text = entry['scenario_text']
                option_A = entry['options']['A']
                option_B = entry['options']['B']
            else:
                scenario_text = entry['scenario_text_mlt'][language]
                option_A = entry['options_mlt'][language]['A']
                option_B = entry['options_mlt'][language]['B']

            prompt = (
                f"{system_prompt}\n"
                f"{scenario_text}\n"
                f"A) {option_A}\n"
                f"B) {option_B}\n\n"
                f"{ending}"
            )
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                outputs = self.model(**inputs)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()
                
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)

            del inputs, outputs, logits, shift_logits, shift_labels, loss, loss_fn
            if DEVICE == "cuda":
                self._cleanup_memory(force=True)
                
        self.model.reset()
        self._cleanup_memory(force=True)
        return np.mean(perplexities)
