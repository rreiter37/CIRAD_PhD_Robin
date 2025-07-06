### Méthodes de détection
	- Les différentes méthodes employées sont sourcées et décrites dans la page [[Biblio]].
	- Pour déterminer si un spectre est outlier ou non avec les méthodes d'autoencodage, on calcule une erreur de reconstruction pour chaque spectre. On obtient donc un vecteur d'erreurs de reconstruction $S_n$, avec $n$ le nombre de spectres du dataset. On peut alors estimer la moyenne $\mu(S_n)$ et l'écart-type $\sigma(S_n)$ de ces erreurs.
		- Une approche simple et classique pour déterminer un seuil de décision est de calculer: $$\mu(S_n) + 2. \sigma(S_n)$$
		  Au-delà de ce seuil, l'erreur est considérée comme révélatrice d'un outlier.
		  
		  On peut cependant critiquer cette approche du fait de la diversité des bases de données spectrales. Certaines bases de données sont très homogènes et ne contiennent a priori pas ou peu d'outliers. D'autres bases de données sont hétérogènes et bruitées.
		  De plus, la taille du dataset est un paramètre influant fortement le seuillage: l'estimation de $(\mu,\sigma)$ n'est pas robuste pour les petites bases de données, et donc le seuil de détection n'est pas fiable.
		  Appliquer le même seuillage pour ces différents cas est contestable et nécessite rectification.
		- Une autre approche est de calculer un seuil de décision en fonction de la taille du dataset: $$\mu(S_n) + \phi(n) . \sigma(S_n), \quad \phi(n) = \frac{\sqrt{n}}{\text{log}(n+2)}$$
		  
		  C'est une approche adaptative en fonction de la taille de la base de données, mais elle ne répond pas vraiment à toutes les attentes énoncées plus haut.
		- Il existe plusieurs autres approches pour faire un seuil adaptatif pertinent. On peut notamment se baser sur l'entropie de la distribution des erreurs de reconstruction. L'entropie quantifie la dispersion d'une distribution. Plus sa valeur est grande, plus la distribution des erreurs est diffuse et plus le seuil doit s'éloigner de la moyenne pour éviter de sur-détecter des outliers. L'entropie corrige indirectement l'hypothèse de normalité: on ne suppose plus que l'écart-type suffit à décrire la dispersion.
		  
		  On peut proposer le seuil suivant: $$\mu(S_n) + \alpha(E). \sigma(S_n), \quad \text{où:} \quad E = -\underset{i}{\sum} p_i \text{log}(p_i), \quad \text{et:} \quad \alpha(E) = \text{log}(1+E)$$
		  
		  La fonction logarithmique permet de modérer l'impact d'une valeur d'entropie très grande, pour éviter que le seuil soit fixé trop bas et soit trop permissif dans le cas de distributions d'erreurs très dispersées.
		  
		  De plus, sa concavité permet un bon contrôle de la sensibilité: les variations des petites valeurs d'entropie sont plus impactantes, alors que les variations des grandes valeurs produisent des effets de seuil de plus en plus faibles (saturation). Ainsi, quand la distribution des erreurs est très concentrée (faible entropie), une variation même minime doit être significative. À l'inverse, quand elle est très dispersée, on évite de surestimer les différences entre erreurs.
- ### Adaptation du modèle aux outliers
	- #### Différencier les types d'outliers (Chandola et al. 2009)
		- Observation unique déviante
		- Anomalies relatives à leur voisinage
		- Ensemble de points aberrants
	- #### Prétraitement spécifique des outliers (Xu et al. 2020)
		- Alignement dynamique (Dynamic Time Warping), ou *baseline correction* spécifique ///// Wasserstein (Lauriane)
		- Autoencodeurs entraînés sur les inliers pour corriger les outliers vers un espace latent plus proche
	- #### Modèles spécialisés pour les outliers (Aytekin et al. 2018; Aggarwal 2016)
		- Un modèle principal entraîné sur les inliers
		- Un sous-modèle ou une correction entraînée spécifiquement sur les outliers, voire avec des techniques de transfert
	- #### Enrichir le modèle par des outlier-aware ensembles (Jacobs et al. 1991)
		- Dans la stack, ajouter un meta-learner qui utilise une distance aux inliers (e.g. Mahalanobis) comme feature additionnelle
		- Introduire un gating mechanism inspiré des Mixtures of Experts
- ### Idées sur l'adaptation aux outliers
	- On commence par différencier les types d'outliers
		- Observation unique déviante -> baseline correction spécifique: lissage de cette anomalie unique. D'après les encadrants, c'est très peu probable que ça arrive avec les données dont on dispose. => Pas besoin de traiter ce cas dans la pipeline !
		- Anomalies relatives à leur voisinage -> récupération du spectre reconstruit par un décodeur? baseline correction?
		- Ensemble de points aberrants -> transport optimal de la distribution des outliers vers la distribution des inliers // entraînement d'un modèle propre à ces outliers dans la stack
	- Comment différencier les types d'outliers ?
		- Faire un clustering sur les scores d'anomalies de reconstruction -> si plusieurs clusters, alors il y a au moins un ensemble de points aberrants
		- Distance de Wasserstein -> sensible aux outliers ayant une distribution différente de celle des inliers, pas juste une observation unique déviante
		- Coefficient $R^2$ entre un spectre reconstruit et son vrai spectre -> si faible, la distribution du spectre est différente de celle des inliers
		- Analyse des résidus locaux -> pour chaque longueur d'onde $i$, on calcule la médiane des valeurs d'absorbance $m_i$. Si $|s_i - m_i| > k.\text{MAD}_i$, où $\text{MAD}_i$ est l'écart absolu médian, alors la valeur est aberrante pour cette longueur d'onde. Si seules quelques longueurs d'onde sont aberrantes, alors le spectre n'est probablement qu'une anomalie locale.