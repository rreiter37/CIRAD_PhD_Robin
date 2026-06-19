# CLAUDE.md — Article_TabPFN

> Contexte spécifique du dossier `Article_TabPFN`. Complète le `CLAUDE.md` racine
> (conventions générales de la thèse), il ne le remplace pas.
> Réponses en **français**, contenu de l'article en **anglais**.

---

## 🎯 Objet du dossier

Rédaction de l'article scientifique benchmarkant **TabPFN** (tabular foundation model)
comme moteur de calibration *training-free* pour la spectroscopie NIR, comparé à des
lignes de base chimiométriques et ML (PLS/PLS-DA, Ridge, CatBoost, CNN-1D).

Benchmark : **66 jeux de données NIR** (54 régression + 12 classification).

---

## 📄 Fichiers du dossier

| Fichier | Rôle |
|---|---|
| `main_latex.tex` | **Version publiée sur arXiv** (référence figée, ne pas modifier sauf demande explicite). |
| `manuscript_ACA.tex` | **Version de travail**, dérivée de `main_latex.tex`, destinée à la soumission dans **_Analytica Chimica Acta_ (ACA)**. C'est le fichier sur lequel on travaille. |
| `all_datasets_overview.md` | Inventaire descriptif des jeux de données du benchmark (sources, tailles, analytes). |

> ⚠️ Pas de dossier de figures local : les figures (`\includegraphics`) pointent vers des
> chemins externes (générées par les scripts de `scripts/Benchmark_tabpfn/get_results/`).
> La compilation LaTeX complète peut donc échouer ici sur les figures manquantes — c'est attendu.

---

## ✍️ Conventions d'édition dans `manuscript_ACA.tex`

Le fichier utilise un système de commentaires et de TODO explicites :

- **`\TODO{...}`** : encadré « À COMPLÉTER / TO BE COMPLETED » affiché dans le PDF.
  Activé par le booléen `showcomments` (préambule, ~l.93, actuellement `true`).
  Passer à `false` avant la soumission finale pour masquer tous les encadrés.
- **Marqueurs `% >>> ACA ... % <<< ACA`** : ajouts/scaffolding spécifiques à ACA.
- **Marqueurs `% >>> ACA-EDIT ... % <<< ACA-EDIT`** : reformulations narratives.
- **Blocs de revert** : chaque édition conserve l'ancienne version en commentaire
  (`% --- ORIGINAL ---`, `% --- PREVIOUS ---`, suivie de `% --- REPLACEMENT ---`).
  → Pour annuler une modif, décommenter l'original et supprimer le replacement.
  → **Toujours préserver ce schéma** lors de nouvelles éditions : commenter l'ancien,
  documenter le pourquoi, garder la possibilité de revenir en arrière.

---

## 📝 Tâches restantes avant soumission

Les `\TODO{}` ouverts dans `manuscript_ACA.tex` (chercher `\TODO` ou `showcomments`) :

1. **Évaluation de l'incertitude des prédictions** — *priorité demandée* (approche finalisée, voir §🔬A)
   - Méthodes : `\label{sec:uncertainty}` (~l.620) → **RÉDIGÉ** (remplace l'ancien `\TODO`).
   - Résultats : `\label{sec:uncertainty_results}` (~l.879) → **à fusionner** dans « Influence of preprocessing »
     (l'analyse vit désormais dans la comparaison Raw↔opt ; cette sous-section autonome devient redondante).
   - Discussion : `\subsection{Predictive uncertainty}` (~l.955) → à rédiger après les résultats.
   - Figure : bloc 2-panneaux **prêt** (snippet LaTeX, voir §🔬A), à insérer une fois le PNG + chiffres produits.
2. **Coût de calcul par modèle** — *priorité demandée*
   - `\subsection{Computational cost}` `\label{sec:cost}` (~l.852) + `Tableau~\ref{tab:computational_cost}` (skeleton à remplir, ~l.856).
3. **Comparison with previously reported results** (~l.906) — *exigence ACA, desk-reject sensitive* :
   situer les performances obtenues vs littérature pour les analytes considérés.
4. **Data / code availability** (~l.1020) : fournir un lien persistant (Zenodo DOI / dépôt public)
   vers scripts, configs et partitions SPXY.
5. **Highlights** : fichier séparé (`highlights`), 3–5 puces ≤ ~85 caractères (squelette en commentaire ~l.124).
6. **Keywords** : 1–7, en anglais (déjà présents ~l.121, à valider).

---

## 🔬 Les deux analyses prioritaires — sources de données et points d'attention

### A. Incertitude des prédictions — APPROCHE FINALISÉE (décidée en session, juin 2026)

**Angle retenu** (après avoir écarté une comparaison inter-modèles par conformal, jugée trop lourde
et hors fil rouge) : intégrer l'incertitude **dans la comparaison TabPFN-Raw ↔ TabPFN-opt** de la
sous-section « Influence of preprocessing ». Question : *le prétraitement améliore-t-il seulement la
précision ponctuelle, ou aussi la qualité de la distribution prédictive native de TabPFN ?*
- **Distribution native uniquement** (pas de conformal, **pas de baselines**, pas de CNN).
- **Régression seule** (54 datasets). Raw vs opt = même modèle pré-entraîné, deux entrées → fair.
- API TabPFN (v6.0.6, vérifiée) : `predict(output_type="full")` → dict `{mean, median, quantiles,
  criterion, logits}`. Quantiles dérivés en **un seul forward pass** via `criterion.icdf(logits, τ)` ;
  NLL via `criterion.forward(logits, y)` ; CRPS via intégrale de pinball sur la grille de quantiles.
- **Métriques** : PICP (couverture), NMPIW (largeur normalisée par σ(y_train)), CRPS normalisé, NLL,
  erreur de calibration (`miscal`), courbe de fiabilité agrégée. Comparaison appariée Raw↔opt par
  **Wilcoxon** (cohérent avec la rigueur stats du reste de l'article).

**⚠️ Provenance des configs TabPFN-opt — PIÈGE VÉRIFIÉ** :
- Les configs optimales viennent **exclusivement de `Results/tabpfn_reg_smart/`** (= ce qu'ingère le
  master ; `best_config_json` du master = 58/58 identiques à `reg_smart`). Clés `shape/scatter/phase2`,
  compatibles avec `build_transformers` du pipeline courant.
- **NE PAS utiliser `Results/tabpfn_reg_final/` ni `tabpfn_reg_final_light/`** : malgré leur nom
  « final », ce sont des runs d'une **version antérieure incompatible** (clés `scaler/baseline/simple/pca`,
  ex. StandardScaler/Gaussian), configs différentes sur les 58 datasets, **non chargeables** par le
  pipeline actuel et **absentes de l'article**. Toujours s'ancrer sur le master.
- Réglages figés par `reg_smart` : **`n_estimators_final = 16`**, checkpoint
  **`tabpfn-v2.5-regressor-v2.5_real.ckpt`** (présent à la **racine du repo**, pas dans `~/.cache/tabpfn/`
  qui n'a que `_default`), seed 42. Uniformiser n_estimators=16 pour Raw **et** opt (isole l'effet préproc).

**Scripts créés** (dans `scripts/Benchmark_tabpfn/get_results/`) :
- `reinfer_tabpfn_distribution.py` : ré-inférence Raw & opt (configs figées) → parquets par sample
  (`y_true, y_pred, y_median, nll, crps, q0.025…q0.975, y_train_std`) dans `Results/tabpfn_uncertainty/preds/`
  + `uncertainty_meta.csv`. Réutilise l'infra de `pipeline_tabpfn_final.py`.
- `compute_uncertainty_tabpfn.py` : agrégation → `uncertainty_datasetwise.csv`, `reliability_curve.csv`,
  `uncertainty_summary.json` (médianes par modèle + tests Wilcoxon appariés).
- `plot_tabpfn_accuracy_reliability.py` : panneau (b), diagramme de fiabilité Raw vs opt (300 dpi),
  sortie par défaut `Figures/reliability_tabpfn_raw_vs_opt.png`.

**Figure (unique, 2 panneaux)** : (a) dumbbell iRMSEP existant (inchangé, source
`Results/analysis/figures/dumbbell_plots_tabpfn/...tabpfn.png`) + (b) reliability diagram. Composition
**côte-à-côte** (dumbbell portrait 0.76 L/H + carré) via `subfigure` (package `subcaption` présent).
Snippet LaTeX prêt (bloc révertible, label `fig:dumbbell_tabpfn_preprocessing` conservé, placeholders
`[[NMPIW_*]]`/`[[nCRPS_*]]` mappés sur `uncertainty_summary.json`). ⚠️ Compromis : le dumbbell (54 labels)
rétrécit de 0.92→0.575\textwidth ; si illisible, régénérer avec polices + grandes ou basculer en Supplementary.

**Références ajoutées** (Methods) : `kuleshov_accurate_2018` (reliability/calibrated regression) et
`gneiting_strictly_2007` (CRPS) — à coller dans `Statistical_tests.bib` (entrées fournies en session).

**Caveats à mentionner côté article** : distribution évaluée *telle quelle* (pas de recalibration) ;
TabPFN tourné en CPU ; splits KS/SPXY *covariate-shifted* → la couverture empirique reflète des
conditions proches du déploiement (à formuler comme un atout).

### B. Coût de calcul
- **Décomposition souhaitée par l'auteur** : (i) phase de validation/recherche de prétraitement,
  (ii) entraînement/ajustement final, (iii) **temps total**, (iv) **temps total pondéré par le
  budget alloué** (normalisation par le budget de recherche d'hyperparamètres — nb de trials/configs).
- **Sources de timing disponibles** :
  - `Results/*/summary_runs.csv` → colonne **`elapsed_sec`** (un temps *par dataset*, mais
    actuellement **search + final fit confondus** au niveau du wrapper `run_*_final.py`, ex.
    `run_tabpfn_final.py` l.190 `t0 = time.time()`).
  - ⚠️ **Lacune connue** : pas de séparation persistée search-phase vs final-fit. Pour obtenir
    (i) et (ii) séparément, instrumenter les pipelines (`pipeline_*_final.py`) ou parser les logs.
- **Matériel** : décrit en `\label{sec:hardware}` (~l.673). **Rappeler que TabPFN a tourné
  uniquement sur CPU** (GPU indisponible) et discuter l'impact sur la comparaison des temps.

---

## 📊 Où vivent les données et les scripts (hors de ce dossier)

- Tables agrégées : `Results/analysis/master/master_results.{csv,parquet}`,
  `Results/analysis/metadata/dataset_metadata.{csv,parquet}`.
  Colonnes clés master : `RMSECV, RMSE_MF, RMSEP, MAE_test, r2_test, preprocessing_pipeline,
  best_config_json, final_predictions_path, ...` (pas de colonne de timing).
- Résultats par expérience : `Results/{tabpfn_*,catboost_*,pls_ridge_*,nicon_*,plsda_*}/`.
- Scripts d'analyse / figures : `scripts/Benchmark_tabpfn/get_results/`
  (build_*, compute_*, plot_*). Les figures de l'article y sont générées.
- Pipelines d'expériences : `scripts/Benchmark_tabpfn/pipeline_*_final.py` + wrappers `run_*_final.py`.
- **Incertitude (cette session)** : scripts `reinfer_tabpfn_distribution.py`, `compute_uncertainty_tabpfn.py`,
  `plot_tabpfn_accuracy_reliability.py` (dans `get_results/`) ; sorties dans `Results/tabpfn_uncertainty/`.
- Métriques principales : **iRMSEP** (RMSEP relatif, PLS = référence) en régression ; **ACCP** en classif.
- Tests statistiques : Friedman + post-hoc Nemenyi, agrégation **au niveau database** (pas dataset),
  diagrammes de différence critique (CD).
- **Figures** : pas de `Figures/` dans `Article_TabPFN/` (compilation Overleaf) ; le manuscrit référence
  `Figures/...` ; les PNG sont générés par `get_results/` (ex. dumbbell sous
  `Results/analysis/figures/dumbbell_plots_tabpfn/`) puis déposés dans le dossier Figures de compilation.

---

## 📌 Exigences propres à _Analytica Chimica Acta_

- Positionnement **obligatoire** vs résultats antérieurs publiés pour les analytes considérés
  (sous-section dédiée, sensible au desk-reject).
- **Highlights** dans un fichier séparé ; **Keywords** en anglais (mots simples, éviter « and »/« of »).
- **Data/code availability** : lien persistant requis.
- Déclaration *competing interests* (formulation Elsevier standard, ~l.999+).

---

## 🎯 Objectif qualité rédactionnelle

L'auteur vise un article **solide, fluide, concis, rigoureux et intéressant** pour ACA.
Lors des éditions : resserrer la prose, éviter les redites, ancrer chaque affirmation sur un
résultat effectivement produit. **Ne reporter aucun résultat avant d'avoir réalisé les expériences
correspondantes** (les `\TODO` le rappellent explicitement). Une narration « fil conducteur »
(operating envelope du moteur training-free : où il aide, comment, où les méthodes classiques
restent préférables) est déjà amorcée — la prolonger plutôt que la contredire.

---

## ⚙️ Règles de travail (rappel)

- Ne **pas** modifier `main_latex.tex` (version arXiv figée) sauf demande explicite.
- Ne **pas** modifier les données de `Data/` ni supprimer de fichiers sans confirmation.
- Toujours **proposer** les commits Git, ne jamais commiter seul.
- API nirs4all / TabPFN incertaine → **lire les sources du clone local**, ne pas supposer.
- En cas de doute sur ce qui est réellement calculé/persisté → vérifier dans `Results/` et les
  scripts avant d'écrire une affirmation méthodologique dans l'article.
