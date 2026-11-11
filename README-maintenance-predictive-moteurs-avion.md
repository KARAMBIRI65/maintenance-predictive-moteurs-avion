# Maintenance prédictive de moteurs d'avion

Prévoir combien de vols il reste à un moteur avant la panne, à partir de ses capteurs.

Les données viennent du jeu C-MAPSS de la NASA, des moteurs simulés.

J'ai comparé une régression Ridge, une forêt aléatoire et XGBoost.

Sur 100 moteurs de test, XGBoost se trompe en moyenne d'environ 11 vols.

Les données sont simulées, les vraies données des constructeurs ne sont pas publiques.

Pour le lancer, installer les dépendances avec `pip install -r requirements.txt` puis ouvrir `notebooks/maintenance_predictive.ipynb`.

Issa Aziz Karambiri
