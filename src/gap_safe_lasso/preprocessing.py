"""Training-only centering and unit-RMS scaling, with a restored intercept."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainingScale:
    mean: np.ndarray
    scale: np.ndarray
    target_mean: float

    @classmethod
    def fit(cls, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if (X.ndim != 2 or min(X.shape) == 0 or y.shape != (X.shape[0],)
                or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y))):
            raise ValueError("expected finite training X (n,p) and y (n,)")
        mean = X.mean(axis=0)
        scale = np.sqrt(np.mean((X - mean) ** 2, axis=0))
        scale[scale == 0] = 1.0
        return cls(mean, scale, float(y.mean()))

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.mean.size or not np.all(np.isfinite(X)):
            raise ValueError("X must be finite with the training feature count")
        return (X - self.mean) / self.scale

    def raw_parameters(self, coef):
        coef = np.asarray(coef, dtype=float)
        if coef.shape != self.scale.shape or not np.all(np.isfinite(coef)):
            raise ValueError("coef must match the training feature count")
        raw = coef / self.scale
        return raw, float(self.target_mean - self.mean @ raw)
