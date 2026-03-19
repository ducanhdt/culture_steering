
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
import numpy as np
from adjustText import adjust_text

def plot_cultural_map(ax, results_df, wvs_data_df=None, additional_vectors_df=None, title="Cultural Map", 
                      legend=True, x_lim=(-1.5, 4.0), y_lim=(-1.5, 3.2), add_texts=True):
    """
    Plots the cultural map with background WVS data and optional model points/vectors.
    """
    # 1. Colors and Base Data
    if wvs_data_df is not None:
        colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#F0E442"]
        unique_categories = wvs_data_df['Category'].unique()
        category_colors = dict(zip(unique_categories, colors[:len(unique_categories)]))

        for category in unique_categories:
            data = wvs_data_df[wvs_data_df['Category'] == category]
            ax.scatter(data['RC1_final'], data['RC2_final'], 
                       c=category_colors.get(category, '#000000'), label=category, alpha=0.4, s=40)
        
        if add_texts:
            texts = []
            for _, row in wvs_data_df.iterrows():
                if x_lim[0] <= row['RC1_final'] <= x_lim[1] and y_lim[0] <= row['RC2_final'] <= y_lim[1]:
                    texts.append(ax.text(row['RC1_final'], row['RC2_final'], row['country.territory'], fontsize=8, alpha=0.6))

    # 2. Additional Points (Model Results)
    if results_df is not None:
        if add_texts and wvs_data_df is None: texts = []
        for _, row in results_df.iterrows():
            color = row.get('color', 'red')
            ax.scatter(row['RC1'], row['RC2'], c=color, marker='X', s=150, zorder=5, edgecolors='black')
            if add_texts:
                if x_lim[0] <= row['RC1'] <= x_lim[1] and y_lim[0] <= row['RC2'] <= y_lim[1]:
                    texts.append(ax.text(row['RC1'], row['RC2'], row['label'], fontsize=10, color=color, fontweight='bold'))

    # 3. Vectors (Arrows for steering)
    if additional_vectors_df is not None and results_df is not None:
        for _, row in additional_vectors_df.iterrows():
            begin_label = row['begin_point_label']
            begin_row = results_df[results_df['label'] == begin_label]
            
            if not begin_row.empty:
                begin_x = begin_row['RC1'].values[0]
                begin_y = begin_row['RC2'].values[0]
                color = row.get('color', 'blue')
                ax.arrow(begin_x, begin_y, row['RC1']-begin_x, row['RC2']-begin_y, 
                        head_width=0.08, head_length=0.08, fc=color, ec=color, 
                        length_includes_head=True, alpha=0.8, zorder=6)
                if add_texts:
                    if x_lim[0] <= row['RC1'] <= x_lim[1] and y_lim[0] <= row['RC2'] <= y_lim[1]:
                        texts.append(ax.text(row['RC1'], row['RC2'], row['label'], fontsize=10, color=color, fontweight='bold'))

    # 4. Final Formatting
    ax.set_xlabel('Survival vs. Self-Expression (X-Axis)')
    ax.set_ylabel('Traditional vs. Secular-Rational (Y-Axis)')
    ax.set_title(title)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.grid(True, linestyle='--', alpha=0.6)

    if add_texts and texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if additional_vectors_df is not None:
            # Custom legend entries for steering methods if they have consistent colors
            steering_methods = additional_vectors_df[['label', 'color']].drop_duplicates()
            for _, sm in steering_methods.iterrows():
                vector_line = mlines.Line2D([], [], color=sm['color'], marker='>', markersize=8, label=sm['label'])
                handles.append(vector_line)
                labels.append(sm['label'])
        
        ax.legend(handles=handles, labels=labels, loc='upper right', fontsize='small', frameon=True)

def plot_distance_deviation(ax, model_points_df, target_country_means, target_countries, title="Distance to Target Culture"):
    """
    Plots a bar chart showing Euclidean distance between model profiles and target WVS country profiles.
    """
    # model_points_df should have 'RC1', 'RC2', 'label'
    # target_country_means: dict {country_name: (RC1, RC2)}
    
    plot_data = []
    for _, row in model_points_df.iterrows():
        label = row['label']
        # Try to find which target country this point refers to (e.g., "Basic: Vietnam" -> "Vietnam")
        target_country = next((c for c in target_countries if c in label), None)
        if target_country and target_country in target_country_means:
            target_rc1, target_rc2 = target_country_means[target_country]
            dist = np.sqrt((row['RC1'] - target_rc1)**2 + (row['RC2'] - target_rc2)**2)
            plot_data.append({'Method': label, 'Distance': dist})
    
    if not plot_data:
        ax.text(0.5, 0.5, "No matching target country found in labels", ha='center', va='center')
        return

    df = pd.DataFrame(plot_data)
    sns.barplot(data=df, x='Method', y='Distance', ax=ax, palette='viridis')
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel('Euclidean Distance')

def plot_perplexity_curve(ax, coefficients, perplexities, label="Model"):
    """
    Plots perplexity vs steering coefficient.
    """
    ax.plot(coefficients, perplexities, marker='o', label=label, linewidth=2)
    ax.set_xlabel('Steering Coefficient')
    ax.set_ylabel('Perplexity')
    ax.set_title('Steering Cost (Fluency)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

def plot_layer_steering_effect(ax, layer_diff_df, layer_id_df, title="Layer-wise Steering Responsiveness"):
    """
    Plots the Top 4 layers with highest differential per question.
    layer_diff_df: Rows=Rank (0,1,2,3), Cols=Questions, Vals=Abs Differential
    layer_id_df: Rows=Rank (0,1,2,3), Cols=Questions, Vals=Layer IDs
    """
    # We transpose to have questions on X axis
    layer_diff_df.T.plot(kind='bar', ax=ax, rot=45, width=0.8)
    
    for i, container in enumerate(ax.containers):
        # Annotate with layer IDs
        # Rank i across all questions
        labels = layer_id_df.iloc[i, :].values
        ax.bar_label(container, labels=labels, padding=3, rotation=0, fontsize=8)
    
    ax.set_ylim(0, max(layer_diff_df.max().max() * 1.3, 0.5))
    ax.set_ylabel('Abs Steering Differential')
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.legend(title="Rank (1st to 4th Responsive)", bbox_to_anchor=(1.05, 1), loc='upper left')
