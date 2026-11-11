"""Métriques : erreur de prévision, score NASA asymétrique, coût de maintenance."""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Coûts relatifs d'une décision de maintenance (ordres de grandeur illustratifs) :
# - rater une panne (le moteur casse en service : arrêt moteur en vol, déroutement,
#   avion cloué au sol, réparation lourde) coûte très cher ;
# - une inspection anticipée inutile coûte une immobilisation de quelques heures.
COUT_PANNE_RATEE = 50
COUT_FAUSSE_ALERTE = 1


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def score_nasa(y, p):
    """Score officiel du challenge PHM 2008 : prévoir **trop tard** est puni plus
    fort que prévoir trop tôt (exp(d/10) contre exp(-d/13)). Plus bas = meilleur."""
    d = np.asarray(p) - np.asarray(y)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def resume_regression(y, p) -> dict:
    return {"RMSE": rmse(y, p), "MAE": float(mean_absolute_error(y, p)),
            "R²": float(r2_score(y, p)), "Score NASA": score_nasa(y, p)}


def cout_decision(y_alerte, alerte) -> float:
    """Coût moyen par moteur d'une règle d'alerte."""
    y_alerte = np.asarray(y_alerte).astype(bool)
    alerte = np.asarray(alerte).astype(bool)
    fn = np.sum(y_alerte & ~alerte)
    fp = np.sum(~y_alerte & alerte)
    return float((COUT_PANNE_RATEE * fn + COUT_FAUSSE_ALERTE * fp) / len(y_alerte))


def meilleur_seuil(y_alerte, score, seuils, alerte_si_inferieur=False):
    """Seuil qui minimise le coût. Pour une RUL prédite, on alerte si RUL <= seuil ;
    pour une probabilité, on alerte si proba >= seuil."""
    couts = []
    for s in seuils:
        alerte = score <= s if alerte_si_inferieur else score >= s
        couts.append(cout_decision(y_alerte, alerte))
    couts = np.array(couts)
    return float(seuils[int(np.argmin(couts))]), couts
