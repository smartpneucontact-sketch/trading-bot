"""Stacked / weighted ensemble wrappers used by the v5 and v6 models.

These classes are referenced by `pickle` when loading the trained model
artifacts. The pipeline's custom unpickler (`_PipelineUnpickler` in
pipeline.py) remaps any `__main__.StackedEnsemble` / training-script
module path to the definitions here so deserialization works regardless
of how the training script was invoked.
"""

from __future__ import annotations

import numpy as np


class StackedEnsemble:
    """Stacked ensemble: base models + Ridge meta-learner (v6).

    `base_models` is a list of `(name, model)` tuples. `meta_model` is a
    scikit-learn linear model (Ridge in practice). `meta_scaler` is an
    optional `StandardScaler` applied to the stacked base preds before
    the meta-learner.
    """

    def __init__(self, base_models: list, meta_model, meta_scaler=None):
        self.base_models = base_models
        self.meta_model = meta_model
        self.meta_scaler = meta_scaler

    def predict(self, X):
        base_preds = np.column_stack([
            model.predict(X) for _, model in self.base_models
        ])
        if self.meta_scaler is not None:
            base_preds = self.meta_scaler.transform(base_preds)
        return self.meta_model.predict(base_preds)

    def predict_with_base(self, X):
        """Return (final_pred, base_preds_dict) for introspection."""
        base_preds: dict = {}
        cols = []
        for name, model in self.base_models:
            preds = model.predict(X)
            base_preds[name] = preds
            cols.append(preds)
        stacked = np.column_stack(cols)
        if self.meta_scaler is not None:
            stacked = self.meta_scaler.transform(stacked)
        return self.meta_model.predict(stacked), base_preds

    @property
    def feature_importances_(self):
        """Importance weighted by Ridge meta-learner absolute coefficients."""
        coefs = np.abs(getattr(self.meta_model, "coef_",
                               np.zeros(len(self.base_models))))
        if coefs.sum() <= 0:
            coefs = np.ones(len(self.base_models)) / len(self.base_models)
        else:
            coefs = coefs / coefs.sum()
        imp = None
        for (_, model), w in zip(self.base_models, coefs):
            fi = getattr(model, "feature_importances_", None)
            if fi is None:
                continue
            fi = np.asarray(fi, dtype=float)
            imp = fi * w if imp is None else imp + fi * w
        return imp

    @property
    def meta_weights(self):
        coefs = getattr(self.meta_model, "coef_", None)
        if coefs is None:
            return {}
        return {name: round(float(w), 4)
                for (name, _), w in zip(self.base_models, coefs)}


class EnsembleModel:
    """Weighted ensemble of sub-models (v5).

    Each entry in `self.models` is `(name, model, weight)`. Prediction is
    the weight-normalized average of sub-model predictions.
    """

    def __init__(self, models: list):
        self.models = models

    def predict(self, X):
        total_weight = sum(w for _, _, w in self.models)
        preds = np.zeros(len(X))
        for _, model, weight in self.models:
            preds += model.predict(X) * weight
        return preds / total_weight

    def predict_with_base(self, X):
        """Return (final_pred, base_preds_dict)."""
        base_preds: dict = {}
        total_weight = sum(w for _, _, w in self.models)
        final = np.zeros(len(X))
        for name, model, weight in self.models:
            p = model.predict(X)
            base_preds[name] = p
            final += p * weight
        return final / total_weight, base_preds

    @property
    def feature_importances_(self):
        """Per-feature importance = weighted sum of sub-model importances."""
        total_weight = sum(w for _, _, w in self.models) or 1.0
        imp = None
        for _, model, weight in self.models:
            fi = getattr(model, "feature_importances_", None)
            if fi is None:
                continue
            fi = np.asarray(fi, dtype=float)
            scaled = fi * (weight / total_weight)
            imp = scaled if imp is None else imp + scaled
        return imp

    @property
    def meta_weights(self):
        """Map sub-model name to its (normalised) ensemble weight."""
        total_weight = sum(w for _, _, w in self.models) or 1.0
        return {name: round(float(w / total_weight), 4)
                for name, _, w in self.models}
