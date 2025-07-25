### Optuna
	- Quand on optimise un modèle de deep learning qui est coûteux en temps de calcul, il faut faire attention à une chose pour ce qui est de la taille des batchs. En effet, une approche classique avec optuna est d'entraîner le réseau avec un nombre plus faible d'epochs pour la recherche d'hyperparamètres afin d'éviter de faire exploser le temps de calcul. C'est une stratégie pertinente, à part si la taille de batch fait partie des hyperparamètres à optimiser, auquel cas un problème se pose. 
	  
	  En effet, dans la phase d'optuna, avec un nombre d'epochs limité à chaque trial, le modèle n'a pas le temps de converger. Cela implique que les petites tailles de batch donneront les meilleurs résultats à chaque essai, car au cours d'une epoch, une petite taille de batch est synonyme d'un grand nombre de mises à jour des poids du modèle, et donc la convergence est la plupart du temps plus rapide (en termes de nombre d'epochs, pas de temps de calcul). Par conséquent, optuna choisira sans cesse les petites tailles de batch dans cette situation où le modèle n'a pas le temps de converger complètement.
	  
	  Comment traiter ce problème ? Une option pertinente est de retirer la taille de batch de la phase d'optimisation optuna, et de la fixer de la façon suivante. On choisit la plus grande taille de batch possible supportée par le GPU. Une grande taille de batch est synonyme de meilleure approximation du gradient, donc la convergence est supposée plus stable. De plus, ça réduit le champ d'exploration des hyperparamètres pour optuna, donc ça améliore le temps de calcul.
	  
	  Néamoins, certains modèles de deep convergent moins bien avec de grandes tailles de batch (e.g. modèles de segmentation, NLP). Pour éviter de surajuster avec un gradient trop "propre", on peut envisager d'ajouter une régularisation dans le modèle (dropout, weight decay).
- ### Hyperband
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