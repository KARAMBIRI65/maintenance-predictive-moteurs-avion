"""Modèles de régression (RUL) et de classification (panne imminente)."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier

SEED = 42


def modeles_regression() -> dict:
    return {
        "Ridge (linéaire)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Forêt aléatoire": RandomForestRegressor(
            n_estimators=150, max_depth=12, min_samples_leaf=5,
            max_features=0.5, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
            reg_lambda=2.0, tree_method="hist", random_state=SEED),
    }


def modeles_classification(y) -> dict:
    poids = (y == 0).sum() / max((y == 1).sum(), 1)
    return {
        "Régression logistique": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, C=0.3, class_weight="balanced")),
        "XGBoost": XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
            scale_pos_weight=poids, tree_method="hist",
            eval_metric="aucpr", random_state=SEED),
    }


def predictions_hors_echantillon(modele, X, y, groupes, n_splits=5, proba=False):
    """Validation croisée **par moteur** : un moteur n'est jamais à la fois
    en entraînement et en validation (sinon le modèle « reconnaît » le moteur)."""
    pred = np.zeros(len(y), dtype=float)
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groupes):
        modele.fit(X.iloc[tr], y.iloc[tr])
        if proba:
            pred[va] = modele.predict_proba(X.iloc[va])[:, 1]
        else:
            pred[va] = modele.predict(X.iloc[va])
    return pred
