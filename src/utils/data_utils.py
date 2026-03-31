
import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
import pickle
import os

class WVSAnalyzer:
    def __init__(self, ivs_data_path="wvs_evs_trend/ivs_data_processed.pkl", country_code_path="data/country_codes.pkl"):
        self.iv_qns = ["A008", "A165", "E018", "E025", "F063", "F118", "F120", "G006", "Y002", "Y003"]
        self.ivs_data = pd.read_pickle(ivs_data_path)
        
        # Initialize and fit FactorAnalyzer
        features = self.ivs_data[self.iv_qns].dropna()
        self.fa = FactorAnalyzer(n_factors=2, rotation='varimax')
        self.fa.fit(features)
        
        # Project original data
        scores = self.fa.transform(features)
        self.ivs_data['RC1'] = np.nan
        self.ivs_data['RC2'] = np.nan
        self.ivs_data.loc[features.index, 'RC1'] = 1.81 * scores[:, 0] + 0.38
        self.ivs_data.loc[features.index, 'RC2'] = 1.61 * scores[:, 1] - 0.01
        
        # Country level means
        self.country_means = self.ivs_data[self.ivs_data['RC1'].notna() & self.ivs_data['RC2'].notna()].groupby(['s003']).agg({
            'RC1': 'mean',
            'RC2': 'mean'
        }).reset_index()
        self.country_means.columns = ['s003', 'RC1_final', 'RC2_final']
        
        # if os.path.exists(country_code_path):
        #     with open(country_code_path, 'rb') as f:
        #         country_codes = pickle.load(f)
        #     # Normalize column names if they differ from expectations
        #     if 'Numeric' in country_codes.columns:
        #         country_codes = country_codes.rename(columns={'Numeric': 's003', 'Country': 'country.territory', 'Cultural Region': 'Category'})
        #     self.country_means = self.country_means.merge(country_codes, on='s003', how='left')
        # else:
            # Fallback if pickle doesn't exist
        csv_path = "data/s003.csv"
        if os.path.exists(csv_path):
            country_codes = pd.read_csv(csv_path)
            # s003.csv columns: "s003","country.territory","Category"
            self.country_means = self.country_means.merge(country_codes, on='s003', how='left')

    def project_scores(self, scores_df):
        """
        Projects a DataFrame of scores (columns matching iv_qns) onto RC1, RC2.
        """
        # Ensure columns are in correct order
        ordered_df = scores_df[self.iv_qns]
        fa_scores = self.fa.transform(ordered_df)
        rc1 = 1.81 * fa_scores[:, 0] + 0.38
        rc2 = 1.61 * fa_scores[:, 1] - 0.01
        return rc1, rc2

    def get_target_country_means(self):
        """
        Returns a dictionary mapping country names to (RC1, RC2) tuples.
        """
        # Using 'country.territory' as the name column from s003 mapping
        if 'country.territory' in self.country_means.columns:
            return self.country_means.set_index('country.territory')[['RC1_final', 'RC2_final']].T.to_dict('list')
        return {}
