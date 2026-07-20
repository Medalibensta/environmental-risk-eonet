"""
Build and execute notebooks/03_environmental_risk.ipynb.

Light EDA runs live; heavy artefacts (hotspots, forecasts, risk index) are
displayed from the CSVs/figures produced by the src/ modules, keeping the
notebook fast and the modelling logic single-sourced in src/.

Run:
    python notebooks/build_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "03_environmental_risk.ipynb"


def md(t): return nbf.v4.new_markdown_cell(t.strip())
def code(t): return nbf.v4.new_code_cell(t.strip())


cells = [
    md("""
# Risques environnementaux — analyse spatio-temporelle (NASA EONET)

**Angle décisionnel** — support à la **gestion de crise et à la résilience** :
identifier les zones à forte exposition aux aléas naturels, comprendre leurs
dynamiques saisonnières et anticiper la fréquence à venir, pour aider au
pré-positionnement des moyens.

**Données** — API **NASA EONET v3** (*Earth Observatory Natural Event Tracker*),
événements naturels géolocalisés et datés (incendies, tempêtes, volcans,
inondations, glace…), agrégés depuis des sources faisant autorité. Ingestion
reproductible paginée par année : **47 327 points d'observation** sur **≈19 500
événements distincts** (2012–2026).

**Plan**
1. Ingestion & structuration spatio-temporelle
2. EDA — dynamiques temporelles, géographie, saisonnalité
3. Hotspots (DBSCAN spatial)
4. Prévision de fréquence (Prophet)
5. Indice de risque composite par zone
6. Conclusions & limites

> Point méthodo clé : un événement = plusieurs points de trajectoire. On compte
> donc les **événements distincts** (pas les points bruts), sinon une tempête
> longuement suivie écraserait des centaines d'incendies ponctuels.
"""),
    code("""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import load_clean

df = load_clean()
events = df.sort_values("date").groupby("event_id", as_index=False).first()
print(f"{len(df):,} points d'observation | {events.event_id.nunique():,} "
      f"événements distincts | {df.year.min()}-{df.year.max()}")
df.head()
"""),
    md("""
## 1. Points d'observation vs événements distincts

La différence est structurante : au niveau des **points bruts**, les tempêtes
dominent (trajectoires longues) ; au niveau des **événements**, ce sont les
incendies (nombreux et ponctuels).
"""),
    code("""
comp = pd.DataFrame({
    "points bruts": df.category_title.value_counts(),
    "événements distincts": events.category_title.value_counts(),
}).fillna(0).astype(int).head(6)
display(comp)
"""),
    md("""
## 2. EDA spatio-temporelle

Volume annuel, concentration géographique par type d'aléa, et saisonnalité.
"""),
    code("""
for fig in ["events_trend.png", "events_per_year.png",
            "category_region_heatmap.png", "seasonality.png"]:
    display(Image(ROOT / "reports" / "figures" / fig, width=780))
"""),
    md("""
**Lecture** — le volume d'événements suivis explose à partir de 2024 (montée en
charge des sources EONET, à ne pas confondre avec une hausse physique des
aléas — voir *Limites*). La heatmap catégorie × région montre des signatures
nettes : incendies concentrés en Amérique du Nord et Afrique, glace de mer aux
pôles. La saisonnalité confirme des cycles annuels marqués (saison des feux,
saison des tempêtes) — ce qui justifie le choix de Prophet en §4.
"""),
    md("""
## 3. Hotspots géographiques (DBSCAN)

DBSCAN sur la métrique **haversine** (eps = 150 km, min_samples = 30) regroupe
les incendies en zones denses récurrentes, sans fixer le nombre de clusters.
Les événements isolés (bruit) sont exclus.
"""),
    code("""
hot = pd.read_csv(ROOT / "reports" / "hotspots.csv")
print(f"{len(hot)} hotspots d'incendies détectés")
display(hot.head(8).round(2))
display(Image(ROOT / "reports" / "figures" / "wildfire_hotspots.png", width=860))
"""),
    md("""
Le plus gros hotspot (>6 000 événements, ouest de l'Amérique du Nord) domine,
suivi de vastes zones en Afrique subsaharienne et dans le nord de l'Australie.
La carte interactive `reports/figures/eonet_hotspots_map.html` permet de zoomer.
"""),
    md("""
## 4. Prévision de fréquence (Prophet)

Séries de comptages **mensuels** d'événements distincts, Prophet avec
saisonnalité annuelle multiplicative, horizon 18 mois.
"""),
    code("""
fc = pd.read_csv(ROOT / "reports" / "forecast_summary.csv")
display(fc.round(0))
display(Image(ROOT / "reports" / "figures" / "forecast_wildfires.png", width=760))
"""),
    md("""
**Validation de bon sens** — le total projeté sur 12 mois est proche du réalisé
des 12 derniers mois pour chaque catégorie (ex. incendies : ~6 700 projetés vs
6 275 réalisés), et l'intervalle à 80 % encadre bien la saisonnalité. Le modèle
capture le cycle annuel, pas des chocs ponctuels — c'est attendu.
"""),
    md("""
## 5. Indice de risque composite par zone

Faute d'un champ de sévérité homogène entre catégories, l'indice combine trois
composantes normalisées par cellule/région :
**fréquence** (log), **récence** (part des événements des 3 dernières années),
**diversité** (nombre de types d'aléas). Pondération 0,5 / 0,3 / 0,2.
"""),
    code("""
region = pd.read_csv(ROOT / "reports" / "risk_index_region.csv")
display(region[["region", "n_events", "n_categories", "recency",
                "risk_index"]].round(3))
display(Image(ROOT / "reports" / "figures" / "risk_index_region.png", width=680))
"""),
    md("""
**Lecture** — l'Afrique ressort en tête (fréquence élevée, forte récence, 12
types d'aléas), devant l'Amérique du Nord (volume) et l'Océanie (récence). La
carte par cellule 5° `reports/figures/risk_index_map.html` donne le détail
géographique. C'est un score d'**exposition relative**, pas une probabilité
absolue.
"""),
    md("""
## 6. Conclusions & limites

**Conclusions**
- Deux familles de techniques combinées : **clustering spatial non supervisé**
  (DBSCAN) + **séries temporelles** (Prophet), plus un indice composite.
- Signatures géographiques et saisonnières nettes ; hotspots stables ;
  prévisions cohérentes avec l'historique récent.

**Limites (essentielles ici)**
- **Biais de couverture** : EONET agrège des sources dont le nombre et la
  granularité augmentent avec le temps. La forte hausse post-2023 reflète
  surtout un **meilleur reporting**, pas nécessairement plus d'aléas. Les
  comparaisons inter-années sont donc à manier avec prudence.
- **Sévérité manquante** : peu de champs magnitude homogènes → l'indice de
  risque pondère fréquence/récence/diversité, pas l'intensité réelle.
- **Régions par bounding boxes** : approximation grossière (pas de reverse-geo
  fin, pas de frontières administratives).
- L'indice est **relatif et illustratif** : à recalibrer avec des données
  d'exposition/vulnérabilité (population, infrastructures) pour un usage réel.
"""),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python",
                              "name": "python3"}
    nb.cells = cells
    print(f"[notebook] executing {len(cells)} cells...")
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(HERE)}}).execute()
    nbf.write(nb, NB_PATH)
    print(f"[notebook] written -> {NB_PATH}")


if __name__ == "__main__":
    main()
