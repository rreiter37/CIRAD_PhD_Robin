## Méthodes d'augmentation
	- ### Ajout de bruit réaliste
		- **Détails:** Ajout d'un bruit blanc ou coloré à l'intensité du spectre (rapport signal sur bruit proche de l'instrument de mesure). On peut adapter la densité spectrale du bruit pour refléter la réalité (hautes et basses fréquences; c.f. perlin/fractal noise).
		- **Apports:** Utile quand il y a de l'overfitting avec Xcal et que les mesures réelles sont bruitées.
		- **Références:** 
		  -> *Ensemble methods and data augmentation by noise addition applied to the analysis of spectroscopic data*, Analytica Chimica Acta [(Sáiz-Abajo et al., 2005)](https://doi.org/10.1016/j.aca.2004.10.086)
		  
		  -> *Tailoring noise frequency spectrum to improve NIR determinations*, Talanta [(Xie et al., 2009)](https://doi.org/10.1016/j.talanta.2009.08.010)
	- ### Translations / pentes / gain
		- **Détails:** Translations verticales, facteurs multiplicatifs, petites pentes (linéaires) pour simuler des différence d'éclairement, de contacts, de pathlength.
		- **Apports:** Améliore la robustesse à des variations instrumentales ou d'échantillonnage.
		- **Références:** 
		  -> *Data Augmentation of Spectral Data for Convolutional Neural Network (CNN) Based Deep Chemometrics*, arxiv.org [(Bjerrum et al., 2017)](https://doi.org/10.48550/arXiv.1710.01927)
		  
		  -> *A Review of Machine Learning for Near-Infrared Spectroscopy*, Sensors [(Zhang et al., 2022)](https://doi.org/10.3390/s22249764)
	- ### Warping spectral / wavelengths shifts / local stretching
		- **Détails:** Déplacer légèrement les positions en longueur d'onde (simuler dérive de calibration) ou appliquer de petites déformations non-linéaires le long de l'axe des longueurs d'onde.
		- **Apports:** Utile lorsque les instruments ou les conditions (e.g. différences thermiques) provoquent des décalages le long de l'axe des longueurs d'onde.
		- **Références:**
		  -> *Data Augmentation of Spectral Data for Convolutional Neural Network (CNN) Based Deep Chemometrics*, arxiv.org [(Bjerrum et al., 2017)](https://doi.org/10.48550/arXiv.1710.01927)
		  
		  -> *Deep learning for near-infrared spectral data modelling: Hypes and benefits*, TrAC Trends in Analytical Chemistry [(Mishra et al., 2022)](https://doi.org/10.1016/j.trac.2022.116804)
	- ### Mixup / interpolation linéaire entre spectres
		- **Détails:** créer un spectre synthétique = α·s_i + (1−α)·s_j (avec analyte interpolé pour la régression, et pondéré pour la classification)
		- **Apports:** Efficace pour régulariser et générer des points intermédiaires, adapté dans le cas de distributions de valeurs continues. Dans le cas de la classification, peut être utilisé pour un rééquilibrage des classes en mixant des échantillons de la même classe ou proches chimiquement.
		- **Référence:** 
		  -> *A Quantitative Spectra Analysis Framework Combining Mixup and Band Attention for Predicting Soluble Solid Content of Blueberries*, Knowledge Science, Engineering and Management: 16th International Conference, KSEM 2023 [(Li et al., 2023)](https://doi.org/10.1007/978-3-031-40292-0_30)
	- ### GAN / WGAN / VAE
		- **Détails:** Entraîner un modèle génératif sur les spectres et générer de nouvelles instances conditionnelles (par classe, ...). Un GAN conditionnel (cGAN) pour reproduire les spectres d'une même classe; ou WGAN pour plus de stabilité. On peut ensuite valider les spectres artificiels avec des indicateurs (PCA, distances, métriques chimiques, ...).
		- **Apports:** Pertinent pour les petits jeux de données, ou fortement déséquilibré. Il faut cependant toujours garder une cohérence chimique avec les spectres générés artificiellement.
		- **Références:** 
		  -> *fNIRS-GANs: data augmentation using generative adversarial networks for classifying motor tasks from functional near-infrared spectroscopy*, Journal of Neural Engineering [(Nagasawa et al., 2020)](10.1088/1741-2552/ab6cb9)
		  
		  -> *Vis–NIR Spectroscopy Combined with GAN Data Augmentation for Predicting Soil Nutrients in Degraded Alpine Meadows on the Qinghai–Tibet Plateau*, Sensors [(Jiang et al., 2023)](https://doi.org/10.3390/s23073686)
	- ### Augmentation avec composantes pures
		- **Détails:** Combiner les profils de composantes pures (modèle linéaire mixte) et en variant les proportions/perturbations pour créer des écarts physico-chimiques plausibles.
		- **Apports:** Utile quand on connaît des spectres de composantes pures ou qu'on peut estimer des signatures locales. Pertinent dans la prédiction de composition.
		- **Référence:** 
		  -> *Generative data augmentation and automated optimization of convolutional neural networks for process monitoring*, Frontiers in Bioengineering and Biotechnology [(Schiemer et al., 2024)](https://doi.org/10.3389/fbioe.2024.1228846)
	- ### Perturbations spectrales localisées (dropout de bandes, masking, jitter)
		- **Détails:** Masquer aléatoirement certaines bandes de longueurs d'onde, ou injecter un jitter (fluctuation du signal) dans certaines bandes. Appliquer un masque binaire sur des segments courts, ou les remplacer par interpolation locale.
		- **Apports:** Améliore la robustesse à des artefacts localisés (capteurs déféctueux, absorptions externes).
		- **Références:**
		  -> *Data Augmentation of Spectral Data for Convolutional Neural Network (CNN) Based Deep Chemometrics*, arxiv.org [(Bjerrum et al., 2017)](https://doi.org/10.48550/arXiv.1710.01927)
		  
		  -> *Deep learning for near-infrared spectral data modelling: Hypes and benefits*, TrAC Trends in Analytical Chemistry [(Mishra et al., 2022)](https://doi.org/10.1016/j.trac.2022.116804)
	- ### Bootstraping / ré-échantillonnage et synthèse via ACP / espaces latents
		- **Détails:** Projeter les spectres sur les composantes principales, perturber les composantes principales dominantes (petites variations sur scores ACP) puis reprojeter.
		- **Apports:** Respecter la variance globale observée tout en générant de nouvelles instances plausibles.
		- **Références:**
		  -> *Data Augmentation of Spectral Data for Convolutional Neural Network (CNN) Based Deep Chemometrics*, arxiv.org [(Bjerrum et al., 2017)](https://doi.org/10.48550/arXiv.1710.01927)
		  
		  -> *Soil data augmentation and model construction based on spectral difference and content difference*, Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy [(Wan et al., 2024)](https://doi.org/10.1016/j.saa.2024.124360)
	- ### SMOTE adapté / oversampling dans l'espace des caractéristiques
		- **Détails:** Appliquer SMOTE sur des vecteurs de caractéristiques extraites (scores PLS/ACP ou features CNN), pas directement sur spectres bruts si non-linéaire.
		- **Apports:** Pertinent pour le rééquilibrage de classes. Il faut cependant vérifier la cohérence physico-chimique du spectre artificiel.
		- **Références:** 
		  -> *Data Augmentation Techniques for Machine Learning Applied to Optical Spectroscopy Datasets in Agrifood Applications: A Comprehensive Review*, Sensors [(Moisés et al., 2024)](https://doi.org/10.3390/s23208562)
		  
		  -> *Imbalanced spectral data analysis using data augmentation based on the generative adversarial network*, Scientific Reports [(Chung et al., 2024)](https://doi.org/10.1038/s41598-024-63285-4)
- ## Limites
	- Les spectres synthétiques peuvent améliorer la généralisation, mais peuvent aussi introduire un biais si l’augmentation n’imite pas fidèlement la physique/les artefacts réels. Toujours valider sur un jeu de test instrumenté séparé. 
	  
	  **Sources:** [(Schiemer et al., 2024)](https://doi.org/10.3389/fbioe.2024.1228846) ; [(Mishra et al., 2022)](https://doi.org/10.1016/j.trac.2022.116804).
- ## Autres sources
	- *Augmenting NIR Spectra in deep regression to improve calibration*, Chemometrics and Intelligent Laboratory Systems [(Wohlers et al., 2023)](https://doi.org/10.1016/j.chemolab.2023.104924)
	- *Exploring Generative Artificial Intelligence and Data Augmentation Techniques for Spectroscopy Analysis*, Chemical Reviews [(Flanagan et al., 2025)](https://doi.org/10.1021/acs.chemrev.4c00815)