- L'hypothèse initiale de la thèse: "Aucun modèle n'est dominant, donc l'assemblage est clé" est fragilisée par l'arrivée d'un modèle généraliste très performant **TabPFN**. Il est donc nécessaire de **rediriger le sujet de thèse**. L'idéal serait de garder une **continuité** avec les premiers mois de travail (benchmark modèles-pp; détection d'outliers; similarité de datasets), tout en intégrant les **SMA** et en restant dans le **domaine NIRS**.
  
  Voici quelques pistes possibles:
- ### Assemblage de modèles -> Assemblage de compétences
	- Au lieu de stacker des modèles généralistes de machine learning (PLS, Ridge, RF, DL, ...), on peut empiler:
		- Des **sources de biais inductifs** (prior spectres NIRS, augmentations, invariances)
		- Des **stratégies de training** (fine-tuning, adaptation par génération de spectres NIRS artificiels, calibration, sélection),
		- De la **signal vision** (représentations et encodage)
		- Des **modèles spécialisés** (PLS robuste, PFN-like, CNN spectral, ...)
		- Des **agents** qui orchestrent tout ça
	- On passe du stacking à un **méta-système**, TabPFN étant un expert central, mais pas le seul à l'oeuvre.
- ### TabPFN pour la NIRS
	- #### Limites du modèle
		- TabPFN est très bon mais il a possiblement quelques **limitations pour la NIRS**:
			- Robustesse aux domain shifts (instrument de mesure, shift lié aux conditions de mesure, Sortir de la gamme de y...)
			- Petits n ou grands p
			- Explicabilité/ Interprétabilité physico-chimique
			- tabulaire: toutes les colonnes interchangeables, profiter d'un encoding en amont?
	- #### Contributions possibles
		- **Représentations adaptées**
			- Encodages adaptés à des signaux quasi continus:
				- RFF (capturer des structures spectrales)
				- embeddings adaptés aux spectres (dérivées, lissage, SNV, MSC, ... mais appris) -> plus pertinent que PCA
				- découpage par bandes spectro + pooling (vision multi-résolution)
				- encodage non supervisé compatible (VAE)
		- **Fine-tuning via génération artificielle de spectres**
			- Générer des spectres plausibles (méthode Greg, méthode TabPFN, ... avec priors physiques)
			- Puis fine-tuning et adaptation de TabPFN
		- **Hyperparamétrisation & calibration**
			- Même si TabPFN est "plug and play", on peut optimiser:
				- stratégies de preprocessing (et pas seulement modèles)
				- feature/token selection
				- calibration (conformal prediction, temperature scaling, ...)
		- #### Robustesse et OOD
			- comparatif NIRS: in-distribution vs OOD (spectro, saison, conditions, ...)
			- Mesure de l'augmentation de l'incertitude et de l'impact sur les performances de passer en OOD
		- #### Toujours un peu de stacking...
			- TabPFN est très bon en moyenne sur tous les datasets, mais il n'est pas forcément optimal tout le temps: TabPFN seul vs TabPFN + CNN ou PLS ou RF ...
			- Sélection dynamique des modèles experts à ajouter en complément:
				- Dataset-level routing: meta-features (p/n, bruit, shift, corrélations, qualité, ...)
				- Sample-level routing: certains spectres “hors-manifold” vont plutôt vers un modèle robuste/physique ou un estimateur d’incertitude.
		- #### Explicabilité du modèle
			- Explicabilité spectrale -> importance par longueurs d'onde, par bandes, par composantes (e.g. baseline vs pics), ...
			- Stabilité des explications: après preprocessings, resampling, ...
			- Comparatif des explications: TabPFN vs PLS ou CNN ou Lasso/Ridge, ...
			- Explications spécifiques: diagnostic d'instrument de mesure, repérage de dérive, détection de contaminants, ...
		- #### Intégration des SMA
			- Développement d'un système intelligent autour de l'expert principal qu'est TabPFN
			  
			  **Agent 1 — Dataset manager**
			- vérifie qualité, détecte drift/outliers, propose splits réalistes (par capteur/instrument)
			  
			  **Agent 2 — Preprocessing strategist**
			- choisit une stratégie de preprocessing par évaluation de métriques sur plusieurs combinaisons
			  
			  **Agent 3 — Model specialist**
			- entraîne TabPFN, PLS, Ridge, LGBM, CNN, ... ; gère calibration; propose incertitudes
			  
			  **Agent 4 — Evaluator**
			- applique protocole robuste (ID/OOD, calibration, stabilité), agrège résultats, signale contradictions
			  
			  **Agent 5 — Explainer**
			- produit explications spectrales + rapport; compare aux connaissances qu'on a déjà (bandes attendues)
			  
			  **Agent 6 — Router (meta-learner)**
			- apprend une politique de sélection/mix d’experts
	- ### Reformulation du sujet de thèse
		- 3 options proposées avec un focus différent à chaque fois:
			- “Adaptation des modèles fondation tabulaires à la spectroscopie proche infrarouge : représentations, robustesse et calibration via systèmes multi-agents.” -> Focus TabPFN
			- “Sélection dynamique et mixtures d’experts pour la NIRS : de l’assemblage de modèles à l’assemblage de compétences autour de **TabPFN à modif**.” -> reste dans la logique stacking en un peu différent
			- “Explicabilité physico-chimique robuste et cohérente des modèles modernes (PFN) en spectroscopie NIRS, orchestrée par systèmes multi-agents.” -> Focus explicabilitié
	- ### Plan de thèse possible
		- **Chapitre Benchmark** : tes pipelines modèle+pp, protocole, déterminisme, heatmaps (j'ai déjà ça)
		- **Chapitre TabPFN** : application aux datasets NIRS + analyse + calibration + limites (OOD)
		- **Chapitre Adaptation NIRS** : encodages / synthèse / fine-tuning / stratégies pp
		- **Chapitre Routing & SMA** : agents, sélection dynamique, mixture of experts, coût/perf
		- **Chapitre Explicabilité** : stabilité, bandes, cohérence physico-chimique, diagnostics