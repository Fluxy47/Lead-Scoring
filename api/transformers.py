import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Stateless transformer replicating all manual cleaning steps.
    Accepts raw DataFrame. Returns cleaned DataFrame with flags created,
    ready for RareCategoryCollapser + ColumnTransformer.
    """

    _rename_map = {
        'Prospect ID': 'prospect_id',
        'Lead Number': 'lead_number',
        'Lead Origin': 'lead_origin',
        'Lead Source': 'lead_source',
        'Do Not Email': 'do_not_email',
        'Do Not Call': 'do_not_call',
        'Converted': 'converted',
        'TotalVisits': 'total_visits',
        'Total Time Spent on Website': 'total_time_on_website',
        'Page Views Per Visit': 'page_views_per_visit',
        'Last Activity': 'last_activity',
        'Country': 'country',
        'Specialization': 'specialization',
        'How did you hear about X Education': 'heard_about_source',
        'What is your current occupation': 'occupation',
        'What matters most to you in choosing a course': 'course_selection_priority',
        'Search': 'search',
        'Magazine': 'magazine',
        'Newspaper Article': 'newspaper_article',
        'X Education Forums': 'x_education_forums',
        'Newspaper': 'newspaper',
        'Digital Advertisement': 'digital_advertisement',
        'Through Recommendations': 'through_recommendations',
        'Receive More Updates About Our Courses': 'receive_course_updates',
        'Tags': 'tags',
        'Lead Quality': 'lead_quality',
        'Update me on Supply Chain Content': 'supply_chain_updates',
        'Get updates on DM Content': 'dm_content_updates',
        'Lead Profile': 'lead_profile',
        'City': 'city',
        'Asymmetrique Activity Index': 'asym_activity_index',
        'Asymmetrique Profile Index': 'asym_profile_index',
        'Asymmetrique Activity Score': 'asym_activity_score',
        'Asymmetrique Profile Score': 'asym_profile_score',
        'I agree to pay the amount through cheque': 'agreed_to_pay_cheque',
        'A free copy of Mastering The Interview': 'free_interview_guide',
        'Last Notable Activity': 'last_notable_activity',
    }

    _binary_cols = [
        'do_not_email',
        'do_not_call',
        'search',
        'magazine',
        'newspaper_article',
        'x_education_forums',
        'newspaper',
        'digital_advertisement',
        'through_recommendations',
        'receive_course_updates',
        'supply_chain_updates',
        'dm_content_updates',
        'agreed_to_pay_cheque',
        'free_interview_guide',
    ]

    _placeholder_cols = [
        'specialization',
        'occupation',
        'city',
        'heard_about_source'
    ]

    _ordinal_remap = {
        1: 3,
        2: 2,
        3: 1
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # 1. Rename columns
        df = df.rename(columns=self._rename_map)

        # 2. Binary Yes/No → int8
        for col in self._binary_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .map({'Yes': 1, 'No': 0})
                    .fillna(0)
                    .astype('int8')
                )

        # 3. Asymmetrique Index
        for col in ['asym_activity_index', 'asym_profile_index']:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.extract(r'^(\d+)')[0]
                    .astype('Int8')
                    .map(self._ordinal_remap)
                    .fillna(0)
                    .astype('int8')
                )

        # 4. total_visits
        if 'total_visits' in df.columns:
            df['total_visits'] = df['total_visits'].fillna(0).astype('int64')

        # 5. Replace placeholder values
        for col in self._placeholder_cols:
            if col in df.columns:
                df[col] = df[col].replace('Select', np.nan)

        # 6. Missingness flags
        for col in [
            'lead_source',
            'total_visits',
            'specialization',
            'occupation',
            'city',
            'heard_about_source'
        ]:
            if col in df.columns:
                df[f'{col}_was_missing'] = df[col].isnull().astype('int8')

        asym_cols = [
            'asym_activity_index',
            'asym_profile_index',
            'asym_activity_score',
            'asym_profile_score'
        ]

        existing_asym = [c for c in asym_cols if c in df.columns]

        if existing_asym:
            df['asym_data_missing'] = (
                df[existing_asym]
                .isnull()
                .any(axis=1)
                .astype('int8')
            )

        # 7. Drop heard_about_source
        if 'heard_about_source' in df.columns:
            df = df.drop(columns=['heard_about_source'])

        return df


class RareCategoryCollapser(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.01, cols=None):
        self.threshold = threshold
        self.cols = cols

    def fit(self, X, y=None):
        self.rare_map_ = {}

        cols = self.cols or X.select_dtypes(include='object').columns

        for col in cols:
            counts = X[col].value_counts(normalize=True, dropna=True)
            self.rare_map_[col] = set(
                counts[counts < self.threshold].index
            )

        return self

    def transform(self, X, y=None):
        X_ = X.copy()

        for col, rare_cats in self.rare_map_.items():
            if col in X_.columns:
                mask = X_[col].isin(rare_cats)
                X_[col] = X_[col].where(~mask, other='Other')

        return X_


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Clips feature values at a fitted upper quantile.
    Fitted on training data only.
    """

    def __init__(self, upper_quantile=0.99):
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        self.upper_limits_ = np.nanquantile(
            np.array(X, dtype=float),
            self.upper_quantile,
            axis=0
        )
        return self

    def transform(self, X, y=None):
        X_ = np.array(X, dtype=float).copy()

        for i, limit in enumerate(self.upper_limits_):
            X_[:, i] = np.clip(X_[:, i], a_min=None, a_max=limit)

        return X_

    def get_feature_names_out(self, input_features=None):
        return input_features