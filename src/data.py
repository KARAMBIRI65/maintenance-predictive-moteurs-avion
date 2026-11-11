"""Chargement des données NASA C-MAPSS et calcul de la durée de vie restante (RUL)."""
from pathlib import Path
import urllib.request
import zipfile
import io

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Signification physique des 21 capteurs (Saxena et al., 2008)
CAPTEURS = {
    "s1": "T2 - température entrée soufflante (°R)",
    "s2": "T24 - température sortie compresseur BP (°R)",
    "s3": "T30 - température sortie compresseur HP (°R)",
    "s4": "T50 - température sortie turbine BP (°R)",
    "s5": "P2 - pression entrée soufflante (psia)",
    "s6": "P15 - pression conduit by-pass (psia)",
    "s7": "P30 - pression sortie compresseur HP (psia)",
    "s8": "Nf - vitesse physique soufflante (tr/min)",
    "s9": "Nc - vitesse physique du cœur (tr/min)",
    "s10": "epr - rapport de pression moteur",
    "s11": "Ps30 - pression statique sortie HPC (psia)",
    "s12": "phi - rapport carburant / Ps30",
    "s13": "NRf - vitesse corrigée soufflante (tr/min)",
    "s14": "NRc - vitesse corrigée du cœur (tr/min)",
    "s15": "BPR - taux de dilution",
    "s16": "farB - rapport carburant/air brûleur",
    "s17": "htBleed - enthalpie de prélèvement",
    "s18": "Nf_dmd - vitesse soufflante demandée",
    "s19": "PCNfR_dmd - vitesse corrigée demandée",
    "s20": "W31 - débit de refroidissement HPT",
    "s21": "W32 - débit de refroidissement LPT",
}
COLONNES = ["moteur", "cycle", "reglage1", "reglage2", "reglage3"] + list(CAPTEURS)

# Plafond de RUL : en début de vie le moteur est sain et sa RUL n'est pas
# observable dans les capteurs. On plafonne donc la cible (Heimes, 2008).
RUL_MAX = 125

URL_NASA = ("https://phm-datasets.s3.amazonaws.com/NASA/"
            "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip")


def telecharger(dest: Path = DATA_DIR) -> None:
    """Télécharge les 4 sous-jeux C-MAPSS depuis le dépôt de la NASA (~12 Mo)."""
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL_NASA) as r:
        outer = zipfile.ZipFile(io.BytesIO(r.read()))
    name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(name)))
    for n in inner.namelist():
        if n.endswith(".txt") and n.split("_")[0] in {"train", "test", "RUL"}:
            (dest / Path(n).name).write_bytes(inner.read(n))


def _lire(fichier: Path) -> pd.DataFrame:
    df = pd.read_csv(fichier, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLONNES)]
    df.columns = COLONNES
    return df


def charger(sous_jeu: str = "FD001"):
    """Renvoie (train, test) avec la colonne RUL.

    - train : chaque moteur tourne jusqu'à la panne, RUL = dernier cycle - cycle.
    - test  : les trajectoires sont coupées avant la panne ; la NASA fournit
              la vraie RUL au dernier cycle observé.
    """
    f_train = DATA_DIR / f"train_{sous_jeu}.txt"
    if not f_train.exists():
        telecharger()
    train = _lire(f_train)
    test = _lire(DATA_DIR / f"test_{sous_jeu}.txt")
    rul_fin = pd.read_csv(DATA_DIR / f"RUL_{sous_jeu}.txt", header=None)[0].to_numpy()

    dernier = train.groupby("moteur")["cycle"].transform("max")
    train["RUL"] = dernier - train["cycle"]

    dernier_test = test.groupby("moteur")["cycle"].transform("max")
    rul_map = dict(zip(range(1, len(rul_fin) + 1), rul_fin))
    test["RUL"] = test["moteur"].map(rul_map) + dernier_test - test["cycle"]
    return train, test


def capteurs_utiles(train: pd.DataFrame, seuil: float = 1e-4) -> list:
    """Garde les capteurs qui varient (les autres sont constants en conditions fixes)."""
    std = train[list(CAPTEURS)].std()
    return std[std > seuil].index.tolist()


def derniers_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par moteur : son dernier cycle observé (protocole officiel du test)."""
    return df.loc[df.groupby("moteur")["cycle"].idxmax()].reset_index(drop=True)
