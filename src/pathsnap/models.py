"""Model zoo for candidate classification. Each entry: family, sklearn-compatible estimator factory, the
configuration used, and an optional row cap for models whose cost grows super-linearly (KNN).

All models output P(candidate is the true arc). Scaling is part of the pipeline (fit inside each CV fold only).
"""
import time
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

SEED = 42


class EpochMLP(BaseEstimator, ClassifierMixin):
    """StandardScaler + sklearn MLP trained epoch by epoch with partial_fit, so we can record the real
    training / validation log loss after every epoch (used for the loss-curve plot)."""

    def __init__(self, hidden=(64, 32), alpha=1e-4, lr=1e-3, batch_size=2048, epochs=20, seed=SEED):
        self.hidden, self.alpha, self.lr, self.batch_size, self.epochs, self.seed = hidden, alpha, lr, batch_size, epochs, seed

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.mlp_ = MLPClassifier(hidden_layer_sizes=self.hidden, alpha=self.alpha, learning_rate_init=self.lr,
                                  batch_size=self.batch_size, random_state=self.seed)
        self.classes_ = np.array([0, 1])
        rng = np.random.default_rng(self.seed)
        mon = rng.choice(len(Xs), min(200_000, len(Xs)), replace=False)   # train-loss monitor subset
        self.history_ = {"epoch": [], "train_log_loss": [], "val_log_loss": []}
        for ep in range(self.epochs):
            perm = rng.permutation(len(Xs))
            for s in range(0, len(Xs), 200_000):              # partial_fit = one pass of mini-batches over a chunk
                idx = perm[s:s + 200_000]
                self.mlp_.partial_fit(Xs[idx], y[idx], classes=self.classes_)
            self.history_["epoch"].append(ep + 1)
            self.history_["train_log_loss"].append(log_loss(y[mon], self.mlp_.predict_proba(Xs[mon])[:, 1], labels=[0, 1]))
            if X_val is not None:
                pv = self.mlp_.predict_proba(self.scaler_.transform(np.asarray(X_val, dtype=np.float32)))[:, 1]
                self.history_["val_log_loss"].append(log_loss(y_val, pv, labels=[0, 1]))
        return self

    def predict_proba(self, X):
        return self.mlp_.predict_proba(self.scaler_.transform(np.asarray(X, dtype=np.float32)))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


MODELS = {
    "LogisticRegression": dict(
        family="Linear", max_rows=None,
        config="StandardScaler + L2 logistic regression, C=1.0, lbfgs, max_iter=500",
        make=lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))),
    "GaussianNB": dict(
        family="Probabilistic", max_rows=None,
        config="Gaussian Naive Bayes, var_smoothing=1e-9 (class-conditional independent Gaussians)",
        make=lambda: GaussianNB()),
    "KNN": dict(
        family="Instance-based", max_rows=150_000,
        config="StandardScaler + distance-weighted KNN, k=25, Euclidean; fit on a 150k-row random subsample",
        make=lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1))),
    "DecisionTree": dict(
        family="Tree", max_rows=None,
        config="CART, gini, max_depth=12, min_samples_leaf=50",
        make=lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=SEED)),
    "RandomForest": dict(
        family="Ensemble (bagging)", max_rows=None,
        config="150 trees, max_depth=None, min_samples_leaf=5, max_features=sqrt, max_samples=0.25",
        make=lambda: RandomForestClassifier(n_estimators=150, min_samples_leaf=5, max_features="sqrt",
                                            max_samples=0.25, n_jobs=-1, random_state=SEED)),
    "LightGBM": dict(
        family="Boosting", max_rows=None,
        config="GBDT, 400 rounds, lr=0.05, num_leaves=63, min_child_samples=50, subsample=0.8, colsample=0.8",
        make=lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                                    n_jobs=-1, verbose=-1)),
    "XGBoost": dict(
        family="Boosting", max_rows=None,
        config="hist trees, 400 rounds, lr=0.05, max_depth=8, subsample=0.8, colsample=0.8",
        make=lambda: XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=8, subsample=0.8,
                                   colsample_bytree=0.8, tree_method="hist", eval_metric="logloss",
                                   random_state=SEED, n_jobs=-1)),
    "MLP": dict(
        family="Neural network", max_rows=None,
        config="StandardScaler + MLP 25-64-32-1, ReLU, Adam lr=1e-3, L2 alpha=1e-4, batch 2048, 20 epochs",
        make=lambda: EpochMLP()),
}


def fit_model(name, X, y, rng=None, **fit_kw):
    spec = MODELS[name]
    if spec["max_rows"] and len(X) > spec["max_rows"]:
        rng = rng or np.random.default_rng(SEED)
        idx = rng.choice(len(X), spec["max_rows"], replace=False)
        X, y = X[idx], y[idx]
    m = spec["make"]()
    t0 = time.perf_counter()
    m.fit(X, y, **fit_kw) if fit_kw else m.fit(X, y)
    return m, time.perf_counter() - t0
