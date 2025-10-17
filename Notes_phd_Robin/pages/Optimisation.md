### Impact de la taille du dataset
collapsed:: true
	- Les résultats sur les bases de données de régression montrent que le modèle de deep learning NICON n'est pas plus performant en général sur les petites et moyennes bases de données que les autres modèles: Ridge, PLS, LGBM. Il est même souvent moins performant. En revanche, sur la base de données de sols LUCAS, qui est de loin la plus grande base de données utilisée dans les calculs, le modèle NICON est bien plus performant. Plusieurs explications sont possibles:
		- **Nombre de paramètres et overfitting:** Les modèles de deep learning ont beaucoup plus de paramètres et donc une plus grande capacité d'approximation. Sur de petits jeux de données, cette capacité est trop élevée par rapport au nombre d'échantillons, ce qui peut mener à de l'overfitting, malgré le dropout, l'early stopping,...
		  Dans ce cas, les modèles linéaires (Ridge, PLS) ou les méthodes à base d'arbres de décision (LGBM) généralisent mieux.
		  Sur la base de données LUCAS, il y a suffisamment d'exemples pour que le réseau apprenne des représentations pertinentes. Le modèle de deep learning tire alors parti de sa flexibilité et devient supérieur.
		- **Nombre de features et complexité du deep learning:** Les méthodes classiques de machine learning (Ridge, PLS) exploitent bien les datasets à faible dimension effective (corrélations linéaires, structures simples). Les modèles de deep, surtout avec des couches de convolution comme dans NICON, sont capables de capturer des motifs locaux et des interactions complexes. Mais cette abilité n'est effective qu'avec assez de données d'apprentissage. C'est pourquoi ces avantages n'apparaissent pas sur les petits datasets.
		- **Optimisation des hyperparamètres:** Sur les petits datasets, l'optimisation par Optuna est plus bruitée (peu de données -> forte variance dans la validation -> instabilité des hyperparamètres). Sur les gros datasets, la validation devient plus fiable, donc Optuna renvoie des hyperparamètres pertinents, ce qui accroit les performances du modèle de deep.
		- **Régularisation implicite:** Les modèles linéaires (Ridge, PLS) ont une régularisation intégrée forte et bien adaptée aux petites tailles (shrinkage des coeffs, réduction de dimension). NICON  utilise Optuna, fait du dropout et de l'early stopping, mais ce sont des régularisations plus souples qui nécessitent plus d'échantillons pour être efficaces.
		- **Biais-Variance:** Modèles classiques (Ridge, PLS) -> biais élevé, variance faible.
		  Deep learning -> biais faible, variance élevée.
		  Avec peu d'échantillons, la variance domine -> NICON moins performant. Avec beaucoup de données, la variance est contrôlée et le faible biais donne un net avantage au deep learning.
		- **Effet batch size / scheduler:** NICON utilise un cosine annealing warm restart. Sur les petits datasets, le nombre d'itérations est trop faible pour exploiter correctement ce scheduler. Sur les gros datasets, l'optimiseur bénéficie pleinement du cycle, ce qui aboutit à une meilleure convergence.
	- **Conclusion:** Le deep learning nécessite un grand nombre de données pour surpasser les méthodes plus simples. Sur des petits datasets, les modèles classiques (Ridge, PLS, LGBM) sont plus robustes et moins sensibles à l'overfitting. Mais dès que le dataset est conséquent, la capacité d'abstraction et la richesse des représentations apprises par le deep prennent le dessus.
- ### Optuna
  collapsed:: true
	- Quand on optimise un modèle de deep learning qui est coûteux en temps de calcul, il faut faire attention à une chose pour ce qui est de la taille des batchs. En effet, une approche classique avec optuna est d'entraîner le réseau avec un nombre plus faible d'epochs pour la recherche d'hyperparamètres afin d'éviter de faire exploser le temps de calcul. C'est une stratégie pertinente, à part si la taille de batch fait partie des hyperparamètres à optimiser, auquel cas un problème se pose. 
	  
	  En effet, dans la phase d'optuna, avec un nombre d'epochs limité à chaque trial, le modèle n'a pas le temps de converger. Cela implique que les petites tailles de batch donneront les meilleurs résultats à chaque essai, car au cours d'une epoch, une petite taille de batch est synonyme d'un grand nombre de mises à jour des poids du modèle, et donc la convergence est la plupart du temps plus rapide (en termes de nombre d'epochs, pas de temps de calcul). Par conséquent, optuna choisira sans cesse les petites tailles de batch dans cette situation où le modèle n'a pas le temps de converger complètement.
	  
	  Comment traiter ce problème ? Une option pertinente est de **retirer la taille de batch de la phase d'optimisation optuna**, et de la fixer de la façon suivante. On choisit **la plus grande taille de batch possible supportée par le GPU**. Une grande taille de batch est synonyme de meilleure approximation du gradient, donc la convergence est supposée plus stable. De plus, ça réduit le champ d'exploration des hyperparamètres pour optuna, donc ça améliore le temps de calcul.
	  
	  Néamoins, certains modèles de deep convergent moins bien avec de grandes tailles de batch (e.g. modèles de segmentation, NLP). Pour éviter de surajuster avec un gradient trop "propre", on peut envisager d'ajouter une régularisation dans le modèle (dropout, weight decay).
	- Quand optuna doit faire sa recherche sur un nombre conséquent d'hyperparamètres, il est pertinent de **réduire intelligemment l'espace de recherche**:
		- Réduire le nombre d'hyperparamètres variables pendant la phase d'optuna. Certains hyperparamètres secondaires (e.g. *activation*, *normalization_method*) peuvent être fixés selon l'expérience que l'on a de l'étude.
		- Il est aussi possible de faire du *conditional sampling* pour activer certains hyperparamètres uniquement selon d'autres.
		- Les *suggest_categorical* d'optuna sont coûteux pour le TPE. Il est intéressant de réduire à 2 choix, ou bien de figer l'hyperparamètre en fonction des résultats obtenus précédemment.
	- Il est possible de **réduire le nombre de ressources allouées par essai**:
		- Réduire le nombre d'epochs associé à la phase de recherche d'optuna, comparé à un plus grand nombre d'epochs pour la calibration du modèle final.
		- Ajouter un *limit_val_batches=p* où p est la proportion du jeu de validation qui sera parcourue pendant la recherche optuna. Idem avec *limit_train_batches=p* pour aller plus loin, en limitant les échantillons parcours dans le jeu d'entraînement.
		- Ajouter un *FastDevRun* pour les premiers essais si la base de données est grande: il s'agit d'un débuggage rapide pour éviter les mauvaises surprises avec un bug en fin d'entraînement après de longs calculs.
	- Plutôt que de reconstruire les tenseurs X et y à chaque essai optuna, on peut les mettre en paramètre de la fonction *objective* pour réutiliser les mêmes TensorDataset pour tous les essais. On perd probablement en robustesse statistique de cette façon en revanche.
	- Réduire la précision numérique: passer en float16 par exemple si on est sur un GPU compatible. Argument *precision* d'un Trainer (e.g. *precision=16*).
	- En complément d'un HyperbandPruner, on peut implémenter un **stepwise training**: réduire le nombre d'epochs sur les premiers essais (e.g. trial 1 = 5 epochs; trial 2 = 10 epochs; trial 3 = 20 epochs; ...). Puis on relance l'étude sur le top n, avec un plus grand nombre d'epochs. Cela permet d'éviter d'entraîner tous les modèles inutilement longtemps.
- ### Hyperband
  collapsed:: true
	- La **technique d'optimisation hyperbande** (ou *Hyperband* en anglais) est une méthode efficace pour la **sélection d'hyperparamètres** en **apprentissage automatique** et **deep learning**. Elle vise à trouver rapidement les meilleures configurations d'hyperparamètres parmi un grand espace de recherche, tout en réduisant le temps de calcul. 
	  Elle repose sur le principe suivant: ne pas entraîner complètement tous les modèles / tester rapidement plein de configurations avec peu de ressources, et n’en entraîner à fond que les plus prometteuses.
	- C'est un **méta-algorithme** qui repose sur deux idées:
		- **Allocation de ressources adaptative** : au lieu d'entraîner tous les modèles jusqu'à la fin, on donne à chacun un petit budget (ex : peu d’epochs, ou un petit sous-ensemble de données), on évalue, et on **élimine les moins bons**.
		- **Successive Halving (réduction successive)** : c’est une stratégie qui consiste à entraîner plusieurs modèles avec peu de ressources, évaluer leurs performances, éliminer une partie (par ex. les 50% moins bons), et redistribuer les ressources restantes aux meilleurs, en augmentant leur budget.
	- Hyperband généralise cette idée en **testant plusieurs stratégies de budget initial vs nombre de configurations**, pour maximiser l’exploration tout en permettant l’exploitation des bons candidats.
	- **Avantages:**
		- Plus rapide qu'un grid search sur de grands espaces;
		- Adaptation automatique à différents "budgets";
		- Pas d'hypothèse forte sur le problème ou le modèle;
		- Palie un coût d'entraînement trop élevé.
	- **Quand l'utiliser:**
		- Beaucoup d'hyperparamètres à explorer;
		- Fort coût d'entaînement d'un modèle (e.g. deep learning);
		- Trouver rapidement un bon modèle.
	- **Implémentations Python:**
		- Optuna: TPE (Tree-structured Parzen Estimator -> optimisation bayésienne) + hyperband
		- Keras tuner: pur hyperband
		- Hyperopt: TPE, annealing
- ### PLS model
  collapsed:: true
	- Comme évoqué plus haut, il est possible lorsqu'on fait plusieurs fois l'évaluation d'un même modèle qu'on hyperparamètre à chaque fois, d'utiliser les résultats précédents pour réduire l'espace de recherche. On peut appliquer cette logique à la PLS: le premier modèle applique une recherche approfondie du nombre optimal de composantes, puis on réduit la grille de recherche à un intervalle restreint du type: $[M(n-1) - \delta; M(n-1) + \delta]$, où $M(n-1)$ est la médiane des composantes optimales des $n-1$ premières évaluations.
	  Cependant, cette approche suppose que les composantes optimales sont toujours proches les unes des autres d'une évaluation à l'autre. Il arrive néanmoins qu'avec cette approche on rate complètement les optima globaux pour les évaluations suivantes car la réduction de l'espace de recherche est trop restrictive.
	  En gardant cette logique d'utilisation des résultats précédents, on peut proposer une approche moins brutale dans la réduction de l'espace de recherche.
		- Les n premiers résultats permettent de construire une médiane $m(n)$ et un écart-type $\sigma(n)$ empiriques, qui définissent une distribution normale $\mathcal{N}(m(n), \sigma(n)^2)$.
		- La distribution est tronquée entre 1 et le nombre maximal de composantes possibles.
		- On peut alors tirer aléatoirement, sans remise, un nombre fini et restreint de nombres de composantes principales à tester dans l'hyperparamétrisation, en suivant cette distribution normale (de façon adaptée au fait que ce sont des entiers).
		- Cette distribution est centrée en la moyenne des résultats précédents, donc on garde la logique précédente. L'écart-type empirique permet par ailleurs de réguler la probabilité d'éloignement à la médiane. C'est-à-dire que si les premiers résultats montrent que les évaluations aboutissent à des nombres de composantes optimaux très différents, alors on favorise les essais d'hyperparamètres loin de la médiane. Au contraire, si les premiers résultats montrent des optima proches, alors on favorise la proximité à la médiane.
- ### Adaptative batch size
	- **Contexte:** Le problème qui se pose quand on calibre des modèles avec différents preprocessings, et sur différents datasets, c'est que la taille de batch optimale est variable. Dans certains cas, les grandes tailles de batch vont permettre de converger beaucoup plus vite avec des gradients qui vont orienter l'optimisation dans la direction du minimum global de façon très efficace. Dans d'autres cas, l'espace d'optimisation de la fonction d'erreur est plus complexe avec par exemple plusieurs minima locaux. Alors des tailles de batch plus petites permettant un grand nombre d'actualisation des poids en une epoch permet plus facilement de sortir des minima locaux pour converger vers un minimum global.
	  
	  La taille de batch optimale dépend donc du dataset. Pour la trouver, une idée est d'utiliser un module d'optimisation comme Optuna, mais cela engendre un biais qui va toujours favoriser les petites tailles de batch (c.f. section Optuna).
	  
	  La solution retenue est alors de fixer la taille de batch comme étant la plus grande acceptée par le GPU. Mais comme on vient de le démontrer, il est parfois meilleur d'utiliser de petites tailles de batch. Une méthode avec une taille de batch adaptative en fonction des données serait de rigueur.
	- #### Estimer le "gradient noise scale" et choisir la taille critique
		- **Principe:** Mesurer l'échelle du bruit du gradient (variance relative) permet d'estimer la plus grande taille de batch utile, au-delà les gains sont marginaux ou il y a même une dégradation. Cette notion de noise scale permet de prédire la plus grande taille de batch utile dans de nombreux domaines. De manière analytique, une formule est donnée pour calculer la plus grande taille de batch qui contrôle le rapport signal sur bruit. #[[Signal to noise ratio]]
		  On peut aussi calculer de façon dynamique, toutes les k itérations, la taille de batch utile au cours de l'entraînement en adaptant dans le même temps le learning rate.
		- **Référence:** 
		  -> *An Empirical Model of Large-Batch Training*, arxiv.org, [McCandlish et al., 2018](https://doi.org/10.48550/arXiv.1812.06162)
	- #### Croissance adaptative du batch pendant l'entraînement (schedules)
		- **Principe:** Démarrer petit (fort bruit utile pour la généralisation), puis augmenter la taille de batch au fil de l'entraînement. C'est équivalent, sous certaines hypothèses, à diminuer le learning rate. Utile lorsque les preprocessings et les datasets modifient la dynamique du gradient.
		- **Référence:** 
		  -> *Don't Decay the Learning Rate, Increase the Batch Size*, ICLR 2018 (conference paper), [Smith et al., 2018](https://doi.org/10.48550/arXiv.1711.00489)
	- #### Adaptation conjointe batch / learning rate (basée sur la variance)
		- **Principe:** Adapter la taille de batch en fonction de la variance observée des gradients et coupler cela au learning rate pour garder un rapport signal/bruit stable. Cela permet de réduire le tuning manuel.
		- **Référence:** 
		  -> *Coupling Adaptive Batch Sizes with Learning Rates*, arxiv.org, [Balles et al., 2017](https://doi.org/10.48550/arXiv.1612.05086)
	- #### Algorithmes d'adaptative sampling / sample-size selection (théorie d'optimisation)
		- **Principe:** Méthode plus théorique qui augmente l’échantillon utilisé pour estimer le gradient quand l’estimation devient trop bruyante (critères de variance). Utile pour garanties de convergence et pour datasets très hétérogènes.
		- **Référence:**
		  id:: 68d25025-65d0-42db-a945-71846caf166a
		  -> *Sample Size Selection in Optimization Methods for Machine Learning*, Mathematical Programming, Series B, vol. 134, p.127-155, [Byrd et al., 2012](https://doi.org/10.1007/s10107-012-0572-5)
	- #### Méthodes d'engineering / heuristiques pratiques
		- **Principe:** Profiler la mémoire et le throughput GPU (max possible), mais multiplier ce maximum par un facteur lié à la *noise scale* (e.g. si noise scale suggère batches 4× plus petits, prendre 1/4 de la taille max). Avec *AdaBatch*: Augmenter le batch automatiquement pour combiner vitesse (grand batch) et convergence (petit batch au début). *AdaBatch* rapporte gains en pratique sur ImageNet/CIFAR.
		- **Référence:** *AdaBatch: Adaptative Batch sizes for training deep neural networks*, arxiv.org, [Devarakonda et al., 2017](https://doi.org/10.48550/arXiv.1712.02029)