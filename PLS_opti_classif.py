from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np

class AutoPLSClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_components=100, cv=5, scale=True, scoring=None):
        self.max_components = max_components
        self.cv = cv
        self.scale = scale
        self.scoring = scoring  # peut être string ou callable

    def fit(self, X, y):
        self.label_binarizer_ = LabelBinarizer()
        y_bin = self.label_binarizer_.fit_transform(y)

        # Assure la bonne forme même en binaire
        if y_bin.ndim == 1 or y_bin.shape[1] == 1:
            y_bin = y_bin.reshape(-1, 1)

        nb_spectra_cv = int(X.shape[0] * (self.cv - 1) / self.cv)
        max_comp = min(self.max_components, X.shape[1], nb_spectra_cv)
        param_grid = {'n_components': list(range(1, max_comp + 1))}

        pls = PLSRegression(scale=self.scale)

        scorer = self.scoring or make_scorer(accuracy_score)

        self.grid_ = GridSearchCV(pls, param_grid, scoring=scorer, cv=self.cv)
        self.grid_.fit(X, y_bin)
        self.best_model_ = self.grid_.best_estimator_
        self.best_n_components_ = self.grid_.best_params_['n_components']

        print(f"[AutoPLSClassifier] Optimal number of components: {self.best_n_components_}")
        return self

    def predict(self, X):
        y_pred = self.best_model_.predict(X)

        if y_pred.shape[1] == 1:
            # Cas binaire : seuil à 0.5
            return self.label_binarizer_.inverse_transform(y_pred > 0.5)
        else:
            # Cas multi-classe : argmax
            return self.label_binarizer_.inverse_transform(np.argmax(y_pred, axis=1))

    def predict_proba(self, X):
        y_pred = self.best_model_.predict(X)

        if y_pred.shape[1] == 1:
            # Clip pour rester dans [0, 1]
            y_prob = np.clip(y_pred, 0, 1)
            return np.hstack([1 - y_prob, y_prob])
        else:
            # Normalisation softmax-like (éventuellement)
            y_pred = np.clip(y_pred, 0, None)
            row_sums = y_pred.sum(axis=1, keepdims=True)
            return y_pred / np.maximum(row_sums, 1e-8)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)