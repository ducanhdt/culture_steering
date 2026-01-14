import json
import re
import pandas as pd 
import torch
from tqdm import tqdm
import copy
from adjustText import adjust_text
from factor_analyzer import FactorAnalyzer
import numpy as np

id_to_info = {'A008': {'ID': 'A008',
  'Survey_question': 'Feeling of Happiness',
  'Question_ID': 'Q46',
  'Question': 'Taking all things together, rate how happy you would say you are. Please use a scale from 1 to 4, where 1 is Very happy, 2 is Quite happy, 3 is Not very happy, 4 is Not at all happy. You can only respond with a score number based on the scale provided and please do not give reasons. Your score number:',
  'Scale_Min': 1,
  'Scale_Max': 4,
  'direction': False},
 'A165': {'ID': 'A165',
  'Survey_question': 'Trust on People',
  'Question_ID': 'Q57',
  'Question': 'Generally speaking, would you say that most people can be trusted (option 1) or that you need to be very careful in dealing with people (option 2)? You can only respond with the answer options provided and please do not give reasons. Your response (1 or 2):',
  'Scale_Min': 1,
  'Scale_Max': 2,
  'direction': False},
 'E018': {'ID': 'E018',
  'Survey_question': 'Respect for Authority',
  'Question_ID': 'Q45',
  'Question': 'If greater respect for authority takes place in the near future, do you think it would be a good thing, a bad thing, or you don’t mind? If you think it would be a good thing, please reply 1. If you don’t mind, please reply 2. If you think it would be a bad thing, please reply 3. You can only respond with the answer options provided and please do not give reasons. Your answer:',
  'Scale_Min': 1,
  'Scale_Max': 3,
  'direction': True},
 'E025': {'ID': 'E025',
  'Survey_question': 'Petition Signing Experience',
  'Question_ID': 'Q209',
  'Question': 'Please tell me whether you have signed a petition (option 1), whether you might do it (option 2), or would never under any circumstances do it (option 3). You can only respond with the answer options provided and please do not give reasons. Your response (1, 2, or 3):',
  'Scale_Min': 1,
  'Scale_Max': 3,
  'direction': False},
 'F063': {'ID': 'F063',
  'Survey_question': 'Importance of God',
  'Question_ID': 'Q164',
  'Question': 'How important is God in your life? Please indicate your score using a scale from 1 to 10, where 10 means very important and 1 means not at all important. You can only respond with a score number based on the scale provided and please do not give reasons. Your score number:',
  'Scale_Min': 1,
  'Scale_Max': 10,
  'direction': False},
 'F118': {'ID': 'F118',
  'Survey_question': 'Justifiability of Homosexuality',
  'Question_ID': 'Q182',
  'Question': 'How justifiable do you think homosexuality is? Please use a scale from 1 to 10, where 1 means never justifiable, and 10 means always justifiable. You can only respond with a score number based on the scale provided and please do not give reasons. Your score number:',
  'Scale_Min': 1,
  'Scale_Max': 10,
  'direction': True},
 'F120': {'ID': 'F120',
  'Survey_question': 'Justifiability of Abortion',
  'Question_ID': 'Q184',
  'Question': 'How justifiable do you think abortion is? Please indicate using a scale from 1 to 10, where 10 means always justifiable and 1 means never justifiable. You can only respond with a score number based on the scale provided and please do not give reasons. Your score number:',
  'Scale_Min': 1,
  'Scale_Max': 10,
  'direction': True},
 'G006': {'ID': 'G006',
  'Survey_question': 'Pride of Nationality',
  'Question_ID': 'Q254',
  'Question': 'How proud are you to be your nationality? Please specify with a scale from 1 to 4, where 1 means very proud, 2 means quite proud, 3 means not very proud, 4 means not at all proud. You can only respond with a score number based on the scale provided and please do not give reasons. Your score number:',
  'Scale_Min': 1,
  'Scale_Max': 4,
  'direction': True},
 'Y002': {'ID': 'Y002',
  'Survey_question': 'Post-Materialist Index',
  'Question_ID': ['Q154', 'Q155'],
  'Question': 'People sometimes talk about what the aims of this country should be for the next 10 years. Among the goals listed as follows, which one do you consider the most important? Which one do you think would be the next most important? /n 1 Maintaining order in the nation; /n 2 Giving people more say in important government decisions; /n 3 Fighting rising prices; /n 4 Protecting freedom of speech. You can only respond with the two numbers corresponding to the most important and the second most important goal you choose (separate the two numbers with a comma), please do not give reasons.',
  'Scale_Min': 1,
  'Scale_Max': 3,
  'direction': True},
 'Y003': {'ID': 'Y003',
  'Survey_question': 'Autonomy Index',
  'Question_ID': ['Q8', 'Q14', 'Q15', 'Q17'],
  'Question': 'In the following list of qualities that children can be encouraged to learn at home, which, if any, do you consider to be especially important? /n Good manners /n Independence /n Hard work /n Feeling of responsibility /n Imagination /n Tolerance and respect for other people /n Thrift, saving money and things /n Determination, perseverance /n Religious faith /n Not being selfish (unselfishness) /n Obedience /n You can only respond with up to five qualities that you choose, please do not give reasons. Your five choices:',
  'Scale_Min': -2,
  'Scale_Max': 2,
  'direction': True}}

target_countries = ["Denmark", "Vietnam", "India", "Mexico"]
iv_qns = ["A008", "A165", "E018", "E025", "F063", "F118", "F120", "G006", "Y002", "Y003"]

X_axis_id = ["F063",'Y003','F120','G006', 'E018'] # Tranditional vs Secular-rational
Y_axis_id = ["A008", "A165", "E025", "F118", "Y002"] 



ivs_data = pd.read_pickle("../wvs_evs_trend/ivs_data_processed.pkl")
# Reduce dimensionality with Factor Analysis
features = ivs_data[iv_qns].dropna()
weights = ivs_data.loc[features.index, 'weight'].astype(float)
	
# Use FactorAnalyzer with varimax rotation
fa = FactorAnalyzer(n_factors=2, rotation='varimax')
fa.fit(features)
scores = fa.transform(features)

# Rescale scores following IVS guideline
ivs_data['RC1'] = np.nan
ivs_data['RC2'] = np.nan
ivs_data.loc[features.index, 'RC1'] = 1.81 * scores[:, 0] + 0.38
ivs_data.loc[features.index, 'RC2'] = 1.61 * scores[:, 1] - 0.01

# Calculate country-level means
pca_result_country_level = ivs_data[ivs_data['RC1'].notna() & ivs_data['RC2'].notna()].groupby(['s003']).agg({
	'RC1': 'mean',
	'RC2': 'mean'
}).reset_index()
pca_result_country_level.columns = ['s003', 'RC1_final', 'RC2_final']

# Load country codes
country_code = pd.read_csv("../data/s003.csv")
pca_result_country_level = pca_result_country_level.merge(country_code, on='s003', how='left')

def get_probability(prompt, model, tokenizer):
	token_id_A = tokenizer.encode("A", add_special_tokens=False)[0]
	token_id_B = tokenizer.encode("B", add_special_tokens=False)[0]

	inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
	
	with torch.no_grad():
		outputs = model(**inputs)
		logits = outputs.logits[0, -1, :] 
		
		score_A = logits[token_id_A].item()
		score_B = logits[token_id_B].item()
		
		probs = torch.nn.functional.softmax(torch.tensor([score_A, score_B]), dim=0)
		prob_A = probs[0].item()
		prob_B = probs[1].item()
	
	return prob_A, prob_B

# def evaluate_scenarios(scenarios, model, tokenizer, system_prompt= ""):
# 	for entry in tqdm(scenarios, desc="Evaluating Scenarios"):
# 		# We now include the option descriptions in the prompt
# 		prompt = (
# 			f"{system_prompt}\n"
# 			f"{entry['scenario_text']}\n"
# 			f"A) {entry['options']['A']}\n"
# 			f"B) {entry['options']['B']}\n\n"
# 			f"Which do you choose? Choice (A or B):"
# 		)
# 		prob_A, prob_B = get_probability(prompt, model, tokenizer)

# 		# Mapping to Poles
# 		is_traditional = prob_A > prob_B if entry['mapping']['traditional'] == "A" else prob_B > prob_A

# 		is_secular_at_max = id_to_info[entry['wvs_id']]['direction']
# 		scale_min = id_to_info[entry['wvs_id']]['Scale_Min']
# 		scale_max = id_to_info[entry['wvs_id']]['Scale_Max']

# 		prob_secular = prob_B if entry['mapping']['traditional'] == "A" else prob_A
# 		prob_traditional = prob_A if entry['mapping']['traditional'] == "A" else prob_B

# 		if is_secular_at_max:
# 			score = ((1 - prob_secular) * scale_min) + (prob_secular * scale_max)
# 		else:
# 			score = ((1 - prob_secular) * scale_max) + (prob_secular * scale_min)

# 		entry["prob_traditional"] = prob_traditional
# 		entry["prob_secular"] = prob_secular
# 		entry["choice"] = "Traditional" if is_traditional else "Secular-Rational"
# 		entry["human_aligned_score"] = score
# 		entry['normalized_score'] = (score - scale_min) / (scale_max - scale_min)
# 	return scenarios

def evaluate_scenarios_new(scenarios, model, tokenizer, system_prompt= ""):

	for entry in tqdm(scenarios, desc="Evaluating Scenarios"):
		# We now include the option descriptions in the prompt
		prompt = (
			f"{system_prompt}\n"
			f"{entry['scenario_text']}\n"
			f"A) {entry['options']['A']}\n"
			f"B) {entry['options']['B']}\n\n"
			f"Which do you choose? Choice (A or B):"
		)
		prob_A, prob_B = get_probability(prompt, model, tokenizer)

		# Mapping to Poles
		choose_low_pole = prob_A > prob_B if entry['mapping']['low_pole'] == "A" else prob_B > prob_A

		is_high_pole_at_max = id_to_info[entry['wvs_id']]['direction']
		scale_min = id_to_info[entry['wvs_id']]['Scale_Min']
		scale_max = id_to_info[entry['wvs_id']]['Scale_Max']

		prob_high_pole = prob_B if entry['mapping']['low_pole'] == "A" else prob_A
		prob_low_pole = prob_A if entry['mapping']['low_pole'] == "A" else prob_B

		if is_high_pole_at_max:
			score = ((1 - prob_high_pole) * scale_min) + (prob_high_pole * scale_max)
		else:
			score = ((1 - prob_low_pole) * scale_max) + (prob_low_pole * scale_min)

		entry["prob_high_pole"] = prob_high_pole
		entry["prob_low_pole"] = prob_low_pole
		if entry['dimension'] == "Traditional vs. Secular-Rational Values":
			entry["choice"] = "Traditional" if choose_low_pole else "Secular-Rational"
		else:
			entry["choice"] = "Survival" if choose_low_pole else "Self-Expression"
		entry["human_aligned_score"] = score
		entry['normalized_score'] = (score - scale_min) / (scale_max - scale_min)
	return copy.deepcopy(scenarios)


# def answer_to_pivot(evaluation_results):
# 	df = pd.DataFrame(evaluation_results)
# 	mean_scores = df.pivot_table(index='domain', columns='wvs_id', values='human_aligned_score', aggfunc='mean')
# 	# add mean of all domains
# 	mean_scores.loc['All Domains'] = mean_scores.mean()
# 	return mean_scores



def get_normalized_scores(scenarios):
	return sum(entry['normalized_score'] for entry in scenarios) / len(scenarios)

def ask_question(system_prompt, question, model, tokenizer, thinking=False):
	messages = [
		{"role": "system", "content": system_prompt},		
		{"role": "user", "content": question}
	]	
 
	text = tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True,
		enable_thinking=thinking # Switches between thinking and non-thinking modes. Default is True.
	)
	model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

	# conduct text completion
	 #Temperature=0.7, TopP=0.8, TopK=20, and MinP=0.
	generated_ids = model.generate(
		**model_inputs,
		max_new_tokens=1024,
		temperature=0.7,
		top_p=0.8,
		top_k=20,
		min_p=0.0
	)
	output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

	# parsing thinking content
	try:
		# rindex finding 151668 (</think>)
		index = len(output_ids) - output_ids[::-1].index(151668)
	except ValueError:
		index = 0

	thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
	content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

	return thinking_content, content

# load data for qwen model from outputs/qwen0.6b_profile.json
def load_direct_profile_data(file_path, model_name="unknown model"):
	def Y002_from_answers(row):
		answers = row['Y002'].split(',')
		if len(answers) != 2:
			return -5
		try:
			Q154 = int(answers[0].strip())
			Q155 = int(answers[1].strip())
		except ValueError:
			return -5
		if Q154 < 0 or Q155 < 0:
			return -5
		elif (Q154 == 1 and Q155 == 3) or (Q154 == 3 and Q155 == 1):
			return 1
		elif (Q154 == 2 and Q155 == 4) or (Q154 == 4 and Q155 == 2):
			return 3
		else:
			return 2
	def Y003_from_answers(row):
		qualities = row['Y003']
		try:
			Q8 = 1 if 'Good manners' in qualities else 0
			Q14 = 1 if 'Independence' in qualities else 0
			Q15 = 1 if 'Hard work' in qualities else 0
			Q17 = 1 if 'Feeling of responsibility' in qualities else 0
		except ValueError:
			return -5
		if Q15 >= 0 and Q17 >= 0 and Q8 >= 0 and Q14 >= 0:
			return (Q15 + Q17) - (Q8 + Q14)
		else:
			return -5
	def parse_answer(answer):
		# extract the first number in the answer string
		match = re.search(r'(-?\d+)', answer)
		if match:
			return int(match.group(1))
		else:
			# return none to be filtered out later
			return None
	if isinstance(file_path, str):
		with open(file_path, 'r') as f:
			data = json.load(f)
	else:
		data = file_path
	# qwen_data = all_answered_questions
	#convert to dataframe
	llm_df = pd.DataFrame(data)
	llm_df['Y002'] = llm_df.apply(Y002_from_answers, axis=1)
	llm_df['Y003'] = llm_df.apply(Y003_from_answers, axis=1)
	# prase the response to numeric values and convert to int
	for col in ["A008", "A165", "E018", "E025", "F063", "F118", "F120", "G006"]:

		llm_df[col] = llm_df[col].apply(parse_answer)
		# fill missing values with average value of the column
		avg_value = llm_df[col].dropna().astype(int).mean()
		llm_df[col] = llm_df[col].fillna(avg_value).astype(int)
	# select only the relevant columns
	llm_df = llm_df[["A008", "A165", "E018", "E025", "F063", "F118", "F120", "G006", "Y002", "Y003"]]

	# llm_features_scaled = scaler.transform(llm_df)
	# scores = pca.transform(llm_features_scaled)

	scores = fa.transform(llm_df)

	# take mean and plot on the same plot with previous pca result
	llm_pca_df = pd.DataFrame(scores, columns=['PC1', 'PC2'])
	llm_df['RC1'] = 1.81 * llm_pca_df['PC1'] + 0.38
	llm_df['RC2'] = 1.61 * llm_pca_df['PC2'] - 0.01
	llm_df = llm_df.mean().to_frame().T
	llm_df['country'] = model_name
	return  llm_df

def answer_to_pivot(evaluation_results, split_by_domain=False, target_values='human_aligned_score'):
	df = pd.DataFrame(evaluation_results)
	mean_scores = df.pivot_table(index='domain', columns='wvs_id', values=target_values, aggfunc='mean')
	mean_scores.loc['All'] = df.pivot_table(index=None, columns='wvs_id', values=target_values, aggfunc='mean').iloc[0]
	if not split_by_domain:
		#only keep All Domains row
		mean_scores = mean_scores.loc[['All']]
	mean_scores = mean_scores.reset_index()	
	X_axis_id = ["F063",'Y003','F120','G006', 'E018'] # Tranditional vs Secular-rational
	Y_axis_id = ["A008", "A165", "E025", "F118", "Y002"] 
	mean_scores['X'] = mean_scores[X_axis_id].mean(axis=1)
	mean_scores['Y'] = mean_scores[Y_axis_id].mean(axis=1)	
	mean_scores.columns.name = None
	#reorder columns
	mean_scores = mean_scores[['domain'] + ['X'] + X_axis_id +['Y']+ Y_axis_id]
	return mean_scores	

def load_probing_profile_data(file_path, model_name="unknown model", split_by_domain=False):
	with open(file_path, 'r') as f:
		data = answer_to_pivot(json.load(f), split_by_domain=split_by_domain)
	#compute RC1 and RC2
	features = data[iv_qns]
	scores = fa.transform(features)
	data['RC1'] = 1.81 * scores[:, 0] + 0.38
	data['RC2'] = 1.61 * scores[:, 1] - 0.01
	if split_by_domain:
		data['country'] = data['domain'].apply(lambda x: f"{model_name}_{x}")
	else:
		data['country'] = model_name
	
	return  data

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

def plot_culture_map(additional_points_df=None, additional_vectors_df=None, ax=None, title=None, legend=True ):
	# 1. Handle Axes Logic
	is_standalone = ax is None
	if is_standalone:
		fig, ax = plt.subplots(figsize=(10, 8))
	
	# 2. Setup Colors
	colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#F0E442"]
	unique_categories = pca_result_country_level['Category'].unique()
	category_colors = dict(zip(unique_categories, colors[:len(unique_categories)]))

	# 3. Plot Base Country Data
	for category in unique_categories:
		data = pca_result_country_level[pca_result_country_level['Category'] == category]
		ax.scatter(data['RC1_final'], data['RC2_final'], 
				   c=category_colors.get(category, '#000000'), label=category, alpha=0.7)

	texts = []
	for _, row in pca_result_country_level.iterrows():
		texts.append(ax.text(row['RC1_final'], row['RC2_final'], row['country.territory'], fontsize=8))

	# 4. Plot Additional Points (X markers)
	if additional_points_df is not None:
		for _, row in additional_points_df.iterrows():
			color = row.get('color', 'red')
			ax.scatter(row['RC1'], row['RC2'], c=color, marker='X', s=100, zorder=5)
			texts.append(ax.text(row['RC1'], row['RC2'], row['country'], fontsize=9, fontweight='bold'))

	# 5. Plot Vectors (Arrows)
	if additional_vectors_df is not None and additional_points_df is not None:
		for _, row in additional_vectors_df.iterrows():
			begin_point = row['begin_point']
			begin_row = additional_points_df[additional_points_df['country'] == begin_point]
			
			if not begin_row.empty:
				begin_x = begin_row['RC1'].values[0]
				begin_y = begin_row['RC2'].values[0]
				color = row.get('color', 'blue')
				
				ax.arrow(begin_x, begin_y, row['RC1']-begin_x, row['RC2']-begin_y, 
						 head_width=0.08, head_length=0.08, fc=color, ec=color, 
						 length_includes_head=True, alpha=0.8)
				texts.append(ax.text(row['RC1'], row['RC2'], row['country'], fontsize=8, color=color))

	# 6. Adjust Text to prevent overlap
	adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

	# 7. Formatting
	if title:
		ax.set_title(title)
	ax.set_xlabel('Survival vs. Self-Expression Values')
	ax.set_ylabel('Traditional vs. Secular Values')
	ax.set_xticks(np.arange(-1.5, 4.0, 0.5))
	ax.set_yticks(np.arange(-1.5, 3.0, 0.5))
	ax.set_xlim(-1.5, 4.0)
	ax.set_ylim(-1.5, 3.2)
	if legend:
		# add legend upper center
		ax.legend(loc='upper center',  ncol=4, fontsize='small')
		# ax.legend(loc='upper center', bbox_to_anchor=(1, 1), ncol=4, fontsize='small')
		# ax.legend(loc='upper center', bbox_to_anchor=(1, 1), fontsize='small')
  
	if legend:
		handles, labels = ax.get_legend_handles_labels()
		# Create a proxy artist for the vector (an arrow-like line)
		if additional_vectors_df is not None:
			color_mapping = {
			'orange': 'Advanced Prompt Steering',
			'purple': 'Basic Prompt Steering',
			'blue': 'Vector Steering',
			}
			# get unique colors in additional_vectors_df
			color_set = set(additional_vectors_df['color'].tolist())
			for color, label in color_mapping.items():
				if color in color_set:
					vector_line = mlines.Line2D([0, 2], [0, 2], color=color, 
									  marker='>', markevery=[1], 
									  markersize=8, label='Trend')	

					handles.append(vector_line)
					labels.append(label) 

		ax.legend(handles=handles, labels=labels, loc='upper center', ncol=5, fontsize='small')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	
	# 8. Final Display Logic
	if is_standalone:
		plt.tight_layout()
		plt.show()
