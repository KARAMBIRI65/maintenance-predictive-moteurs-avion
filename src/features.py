"""Variables explicatives construites uniquement à partir du passé de chaque moteur.

Chaque fenêtre glissante ne regarde que les cycles déjà observés :
aucune information du futur ne fuit dans les variables (pas de « fuite de données »).
"""
import numpy as np
import pandas as pd


def _pente(serie: pd.Series, fenetre: int) -> pd.Series:
    """Pente de la droite des moindres carrés sur les `fenetre` derniers cycles."""
    t = pd.Series(np.arange(len(serie), dtype=float), index=serie.index)
    cov = serie.rolling(fenetre, min_periods=3).cov(t)
    var = t.rolling(fenetre, min_periods=3).var()
    return cov / var


def construire(df: pd.DataFrame, capteurs: list, fenetres=(5, 20)) -> pd.DataFrame:
    """Ajoute, pour chaque capteur : moyenne glissante, écart-type, pente, écart au départ."""
    out = [df[["moteur", "cycle"]]]
    g = df.groupby("moteur", group_keys=False)
    nouvelles = {}
    for c in capteurs:
        x = df[c]
        nouvelles[c] = x
        court, long_ = fenetres
        nouvelles[f"{c}_moy{court}"] = g[c].transform(lambda s: s.rolling(court, min_periods=1).mean())
        nouvelles[f"{c}_moy{long_}"] = g[c].transform(lambda s: s.rolling(long_, min_periods=1).mean())
        nouvelles[f"{c}_std{long_}"] = g[c].transform(lambda s: s.rolling(long_, min_periods=2).std()).fillna(0)
        nouvelles[f"{c}_pente{long_}"] = g[c].transform(lambda s: _pente(s, long_)).fillna(0)
        # dérive par rapport aux 10 premiers cycles du moteur (son « état neuf »)
        ref = g[c].transform(lambda s: s.iloc[:10].mean())
        nouvelles[f"{c}_derive"] = nouvelles[f"{c}_moy{court}"] - ref
    out.append(pd.DataFrame(nouvelles, index=df.index))
    return pd.concat(out, axis=1)


def colonnes_X(features: pd.DataFrame) -> list:
    return [c for c in features.columns if c != "moteur"]
