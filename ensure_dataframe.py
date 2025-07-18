from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class EnsureDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
        else:
            self.feature_names = [f"f{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self.feature_names)
