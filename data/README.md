# Données NASA C-MAPSS

Fichiers inclus (sous-jeux FD001 et FD003, une seule condition de vol) :

| Fichier | Contenu |
|---|---|
| `train_FD00x.txt` | trajectoires complètes jusqu'à la panne (100 moteurs) |
| `test_FD00x.txt` | trajectoires coupées avant la panne (100 moteurs) |
| `RUL_FD00x.txt` | vraie durée de vie restante de chaque moteur de test |

Colonnes : n° moteur, cycle, 3 réglages de vol, 21 capteurs (voir `src/data.py` pour leur signification).

Pour récupérer les 4 sous-jeux (FD002 et FD004 : six conditions de vol), lance depuis la racine :

```bash
python -c "from src.data import telecharger; telecharger()"
```

Source : NASA Prognostics Center of Excellence, *Turbofan Engine Degradation Simulation Data Set*.
Référence : A. Saxena, K. Goebel, D. Simon, N. Eklund, « Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation », PHM 2008.
