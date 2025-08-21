## Detection of outliers [[Outlier Detection]]
collapsed:: true
	- ### Outlier detection with data depth theory
	  link:: https://www.tandfonline.com/doi/epdf/10.1080/1573062X.2017.1280515?needAccess=true
	  title:: Outlier detection in UV/Vis spectrophotometric data
	  author:: Lepot et al.
	  date:: 2017
	  journal:: Urban Water Journal
	  topics:: Outlier detection, Spectroscopic data
	  collapsed:: true
		- Proposal of two techniques of outlier detection in UV/Vis spectroscopic data:
			- **Method 1 : PCA** performed on the **centered** data set, and computation of scores for each spectrum. Then are considered outliers spectra with **PC1 scores** outside the interval defined by the estimated mean of scores $\mu$, and by the estimated standard deviation $\sigma$: $\bm{\mu \pm 2 \sigma}$.
			  
			  -------
			- **Method 2 :**  A first approach based on **Data Depth Theory**. Let A be defined as the matrix of absorbances of size $N_T \times n_x$, containing $N_T$ recorded spectra. Each spectrum measures $n_x$ wavelengths.
			  
			  For a given spectrum j, we compute the Euclidian distance: 
			  $$ED_j = \frac{1}{N_T} \sqrt{\underset{i=1}{\overset{n_x}{\sum}}(A_{j,i} - A_{k \neq j, i})^2}$$
			  Then a threshold is defined to detect an outlier if the Euclidian distance associated to a spectrum is considered too high:
			  $$ED_j > k_M \times median([ED_1:ED_{N_T}])$$
			  With $k_M$ a multiplicative coefficient, hyperparameter of the method. The detection of outliers is therefore sensitive to the subjective value of $k_M$.
			  --------
			- **Method 2 bis :** In order to increase the objectivity and the robustness of this first approach, a complementary method is proposed. After removing from the spectral data set the first outliers detected with the latter approach based on Euclidian distances, we compare the positions of spectra to find the **Most Representative Spectrum (MRS)**.
			  
			  For each wavelength $i$, the spectra with a higher/equal/lower than in spectrum j are counted, and those counts are stored in vectors *Lower, Equal, Higher* respectively.
			  Then the difference between the number of higher and lower absorbances is computed and stored in the vector *Diff*. $$\text{Diff} = |L_1 - L_3|$$
			  
			  This is repeated for every wavelength i and every spectrum j to create the matrices *DIFF* and *EQUAL*. Those matrices are summed over the wavelenghts into the column vectors $S_{DIFF}$ and $S_{EQUAL}$. Then the MRS is identified as the one that maximizes $S_{EQUAL}$ amongst the spectra that minimize $S_{DIFF}$.
			  
			  A measure of dissimilarity between each spectrum and the MRS can be used (e.g. the Mahalanobis distance):
			  $$d_j = \| X_j - X_{MRS} \|$$
			  If the distance is abnormally high, the associated spectrum is considered an outlier.
			  
			  The way this is used to predict outliers with the first approach is not explicitely described in the article. An idea could be to apply the first approach for diverse values of $k_M$ to find first outliers, then estimate the MRS on the filtered data set as the most recurrent spectrum found.
	- ### Outlier detection with space transformation
	  link:: https://pure.au.dk/ws/files/68749533/SDM2013.pdf
	  title:: Outlier Detection with Space Transformation and Spectral Analysis
	  author:: Dang et al.
	  date:: 2013
	  journal:: Proceedings of the 2013 SIAM International Conference on Data Mining
	  topics:: Outlier detection, non parametric, space transformation
	  collapsed:: true
		- #### 1 — Objective & Context
		  
		  **Objective** : Identify **outliers** (globals or locals) with no hypothesis on the data distribution.
		  
		  **Problems of classical methods :**
			- Sensitive to non convex forms
			- Unefficient with varying densities
		- **General idea :**
		  
		  1. Transformation of the space of the data (via a graph)
		  
		  2. Adaptative weighting with local entropy
		  
		  3. Spectral analysis to detect the outliers
		  
		  ---
		- #### 2 — Graph & Local Entropy
			- **Graph G = (V, E)** :
				- Vertices = spectra
				- Edges = closest spectra (ℓ-nearest neighbors)
			- **Weight matrix** :
			  :LOGBOOK:
			  CLOCK: [2025-04-25 Fri 09:29:11]--[2025-04-25 Fri 09:29:11] =>  00:00:00
			  CLOCK: [2025-04-25 Fri 09:29:12]--[2025-04-25 Fri 09:29:12] =>  00:00:00
			  :END:
			  
			  $K_{ij} = \exp\left(-\frac{\|x_i - x_j\|^2}{2(\beta \sigma)^2}\right)$
			- **Local Quadratic Entropy :**
			  
			  $QE(x) = -\log\left( \frac{1}{(\ell+1)^2} \sum_{p,q} G(x_p - x_q, 2\sigma^2) \right), \quad$
			  
			  $\beta(x_i, x_j) = \frac{\\min(QE(x_i), QE(x_j))}{\max(QE(x_i), QE(x_j))}$
			  
			  ---
		- #### 3 — Spectral Analysis
			- **Function to minimize :**
			  
			  $Y^* = \underset{Y}{\text{argmin}}\sum_i \sum_j \|y_i - y_j \|^2 K_{ij}$
			- **Laplacian of the graph :**
			  $L = D - K \quad \text{with } D_{ii} = \sum_j K_{ij}$
			- **Spectral optimization :**
			  $\min_f f^T L f \quad \text{avec } f^T f = 1, \quad f = Y^{T}$
			- **Problem of determination of eigenvalues :**
			  $Lf = \lambda f$
				- Valeurs nulles → global outliers
				- Significant shift in eigenvalues → local outliers
				  
				  ---
		- #### 4 — Outlier detection
		- ##### Global:
			- Low connectivity → λ ≈ 0
			- Extreme values in a single eigenvector
		- ##### Local:
			- Subtle anomalies inside a low connected component
			- Detected by discontinuities in the second eigenvector $f_2$
			  
			  ---
		- #### 5 — Results & Advantages
			- **Advantages:**
				- No hypothesis on the distributions
				- Robustness to complex forms and varying densities
			- **Complexity :**
			  $\mathcal{O}(DN \log N)$
			  
			  ---
	- ### Outlier detection using robust Mahalanobis distance
	  link:: https://www.sciencedirect.com/science/article/pii/S0098300404002304#fig5
	  title:: Multivariate outlier detection in exploration geochemistry
	  author:: Filzmoser et al.
	  date:: 2005
	  journal:: Computers & Geosciences
	  topics:: Outlier detection, Mahalanobis distance, robust distance, adaptative threshold
	  collapsed:: true
		- This approach makes use of the **Mahalanobis distance** to compute a distance of a spectrum to the "center" of the whole data set. A common problem with this approach is that the Mahalanobis distance is very sensitive to the presence of outliers. A robust estimate of the covariance matrix and of the center of the data set is therefore necessary. 
		  
		  The paper provides a method that aims to get a **robust estimate** of the Mahalanobis distance for every spectrum in the data set. For instance, in order to estimate the center of the data set $\mathcal{X}$, it is proposed to compute the centroid from the subset $\mathcal{X_s}$ of fixed size $h$, that minimizes the determinant of the sample covariance matrix. $h$ is here defined as $0.75 n$, where $n$ is the sample size. **Multivariate quantiles**  can then be defined as the points whose Mahalanobis distance is equal to the quantile of the distribution of all Mahalanobis distances.
		  
		  Nevertheless, the approach requires the **normality** of every variable in the data set, in order to approximate the distribution of the Mahalanobis distances as a chi-squared distribution. The hypothesis tests performed revealed that **the condition of normality is not realistic**. To apply the method anyways, it might be possible to find another approximation of the distances' distribution. It would however lead to higher biases in the estimates since the approximate would not be as precise as the one proposed by the authors.
	- ### Outlier Detection with deep learning techniques on time series
	  link:: https://dl.acm.org/doi/10.1145/3691338
	  title:: Deep Learning for Time Series Anomaly Detection: A Survey
	  author:: Darban et al.
	  date:: 2024
	  journal:: ACM Computing Surveys, Volume 57, Issue 1
	  topics:: Outlier detection, Deep Learning, time series
	  collapsed:: true
		- This paper presents the state of the art **time series anomaly detection (TSAD)** with an approach based on deep learning. Both cases of univariate and multivariate time series are treated in this article. We will focus on univariate time series since our spectra are unidimensional.
		- The proposed techniques use diverse structures including RNN, HTM (Hierarchical Temporal Memory), CNN, VAE, AE. These can be whether unsupervised, semi-supervised or supervised. Semi-supervised means it requires laballed normal data, unlike unsupervised methods that require a fully labelled dataset of both normal and anomalous points.
		  In our case, it might be better to focus on unsupervised or semi-supervised since NIRS does not give information about abnormal spectra. It lets us 11 different methods to explore in the litterature, including 4 RNN, 2 HTM, 1 CNN, 3 VAE and 1 AE.
		- #### RNN
		  collapsed:: true
			- ### Anomaly detection with LSTM Neural Networks
			  link:: https://www.semanticscholar.org/paper/Unsupervised-Anomaly-Detection-With-LSTM-Neural-Ergen-Kozat/898a12f14553bf5d5cb18458719b963c14bb81c8
			  title:: Unsupervised and Semi-supervised Anomaly Detection with LSTM Neural Networks
			  author:: Ergen et al.
			  date:: 2019
			  journal:: IEEE Transactions on Neural Networks and Learning Systems
			  topics:: Outlier detection, Deep Learning, LSTM, RNN, time series
			  collapsed:: true
			- ### LSTM for anomaly detection
			  link:: https://www.researchgate.net/publication/304782562_Long_Short_Term_Memory_Networks_for_Anomaly_Detection_in_Time_Series
			  title:: Long Short Term Memory Networks for Anomaly Detection in Time Series
			  author:: Malhotra et al.
			  date:: 2015
			  journal:: ESANN
			  topics:: Outlier detection, Deep Learning, LSTM, RNN, time series
			- ### LSTM for anomaly detection in ECG time series
			  link:: https://www.researchgate.net/publication/308852664_Anomaly_detection_in_ECG_time_signals_via_deep_long_short-term_memory_networks
			  title:: Anomaly detection in ECG time signals via deep long short term memory networks
			  author:: Chauhan et Vig
			  date:: 2015
			  journal:: IEEE International Conference on Data Science and Advanced Analytics (DSAA)
			  topics:: Outlier detection, Deep Learning, LSTM, RNN, time series
			  collapsed:: true
			- ### Collective outlier detection with LSTM
			  link:: https://www.researchgate.net/publication/309370951_Collective_Anomaly_Detection_Based_on_Long_Short-Term_Memory_Recurrent_Neural_Networks
			  title:: Collective Anomaly Detection Based on Long Short-Term Memory Recurrent Neural Networks
			  author:: Bontemps et al.
			  date:: 2016
			  journal:: Future Data and Security Engineering (FDSE conference)
			  topics:: Outlier detection, Deep Learning, LSTM, RNN, time series
			  collapsed:: true
			- ### Outlier detection with Bi-LSTM
			  link:: https://www.researchgate.net/publication/369368513_A_Bi-LSTM_Autoencoder_Framework_for_Anomaly_Detection_--_A_Case_Study_of_a_Wind_Power_Dataset
			  title:: A Bi-LSTM Autoencoder Framework for Anomaly Detection -- A Case Study of a Wind Power Dataset
			  author:: Raihan & Ahmed
			  date:: 2023
			  journal:: IEEE 19th Conference on Automation and Engineering (CASE)
			  topics:: Outlier detection, Deep Learning, LSTM, RNN, time series
			  collapsed:: true
		- #### HTM
		  collapsed:: true
			- ### Real-time anomaly detection with HTM
			  link:: https://www.sciencedirect.com/science/article/pii/S0925231217309864
			  title:: Unsupervised real-time anomaly detection for streaming data
			  author:: Ahmad et al.
			  date:: 2017
			  journal:: Neurocomputing (volume 262, pages 134-147)
			  topics:: Outlier detection, Deep Learning, HTM, time series
			  collapsed:: true
			- ### Anomaly detection with HTM
			  link:: https://www.sciencedirect.com/science/article/pii/S0925231217313887
			  title:: Hierarchical Temporal Memory method for time-series-based anomaly detection
			  author:: Wu et al.
			  date:: 2018
			  journal:: Neurocomputing (volume 273, pages 535-546)
			  topics:: Outlier detection, Deep Learning, HTM, time series
			  collapsed:: true
		- #### CNN
		  collapsed:: true
			- ### Anomaly detection with CNN
			  link:: https://arxiv.org/pdf/1906.03821
			  title:: Time-Series Anomaly Detection Service at Microsoft
			  author:: Ren et al.
			  date:: 2019
			  journal:: Association for Computing Machinery
			  topics:: Outlier detection, Deep Learning, CNN, time series
			  collapsed:: true
		- #### VAE
		  collapsed:: true
			- ### Anomaly detection with VAE for Seasonal KPIs
			  link:: https://dl.acm.org/doi/pdf/10.1145/3178876.3185996
			  title:: Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications
			  author:: Xu et al.
			  date:: 2018
			  journal:: International World Wide Web Conferences Steering Committee
			  topics:: Outlier detection, Deep Learning, VAE, time series
			  collapsed:: true
			- ### Anomaly Detection with conditional variational autoencodeur
			  link:: https://www.researchgate.net/publication/333072648_Robust_and_Unsupervised_KPI_Anomaly_Detection_Based_on_Conditional_Variational_Autoencoder
			  title:: Robust and Unsupervised KPI Anomaly Detection Based on Conditional Variational Autoencoder
			  author:: Li et al.
			  date:: 2018
			  journal:: IPCC Conference
			  topics:: Outlier detection, Deep Learning, VAE, time series
			  collapsed:: true
			- ### Anomaly Detection with Adversarial Training of VAE
			  link:: https://www.researchgate.net/publication/333851045_Unsupervised_Anomaly_Detection_for_Intricate_KPIs_via_Adversarial_Training_of_VAE
			  title:: Unsupervised Anomaly Detection for Intricate KPIs via Adversarial Training of VAE
			  author:: Chen et al.
			  date:: 2019
			  journal:: IEEE Conference on Computer Communications INFOCOM
			  topics:: Outlier detection, Deep Learning, VAE, time series
			  collapsed:: true
		- #### AE
		  collapsed:: true
			- ### Anomaly detection with AE
			  link:: https://www.researchgate.net/publication/304758073_LSTM-based_Encoder-Decoder_for_Multi-sensor_Anomaly_Detection
			  title:: LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection
			  author:: Malhotra et al.
			  date:: 2016
			  journal:: 
			  topics:: Outlier detection, Deep Learning, AE, time series
			  collapsed:: true
	- ### Using the architecture of transformers
	  collapsed:: true
		- #### Founding article on the architecture of transformers
		  link:: https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
		  title:: Attention Is All You Need
		  author:: Vaswani et al.
		  date:: 2017
		  journal:: Computing Research Repository (CoRR)
		  topics:: Outlier detection, Deep Learning, Transformers, attention mechanisms
		  collapsed:: true
			- This paper presents the architecture of Transformers for transduction tasks. It consists of using Multi-Head Attention layers in both encoder and decoder processes. It allows a less sequential structure, and subsequently the possibility to parallelize tasks during the training phase, thus it reduces the computation time compared to sequential and convolutional layers.
		- ---
		- #### Anomaly Transformer
		  link:: https://arxiv.org/abs/2110.02642
		  title:: Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy
		  author:: Xu et al.
		  date:: 2022
		  journal:: CoRR
		  topics:: Outlier detection, Deep Learning, Transformers, time series
		  collapsed:: true
			- Ce modèle introduit le concept de *assocation discrepancy* en exploitant les poids d'attention pour identifier les anomalies. L'idée principale est que les points anormaux ont des associations faibles avec le reste de la série, ce qui les rend détectables via une attention auto-référencée.
		- ---
		- #### W-Transformer used for prediction of time series
		  link:: https://arxiv.org/abs/2209.03945
		  title:: W-Transformers : A Wavelet-based Transformer Framework for Univariate Time Series Forecasting
		  author:: Sasal et al.
		  date:: 2022
		  journal:: 2022 21st IEEE International Conference on Machine Learning and Applications (ICMLA)
		  topics:: Prediction, Deep Learning, Transformers, time series
		  collapsed:: true
			- Ce modèle combine la transformation en ondelettes discrètes à recouvrement maximal (MODWT) avec des Transformers locaux pour capturer les dépendances non stationnaires et non linéaires à long terme dans les séries temporelles univariées.
			  -> **Attention cet article évoque la prédiction de séries temporelles mais pas la détection d'anomalies!**
		- ---
		- #### Stacked Transformer representation & one-dimensional convolutional network (STOC)
		  link:: https://www.sciencedirect.com/science/article/pii/S0952197623001483
		  title:: Time-series anomaly detection with stacked Transformer representations and 1D convolutional network
		  author:: Kim et al.
		  date:: 2023
		  journal:: Engineering Applications of Artificial Intelligence
		  topics:: Outlier detection, Deep Learning, Transformers, time series
		  collapsed:: true
			- **Résumé rapide :** Cette méthode non supervisée empile les représentations de chaque couche d'un encodeur basé sur l'architecture d'un Transformer et utilise dans le decodeur une couche de convolution 1D pour fusionner ces représentations, permettant ainsi de capturer à la fois les tendances globales et les variations locales des séries temporelles.
			- ---
			- **Résumé détaillé :**
				- **Objectif:** L’article propose une méthode d’apprentissage **non supervisé** pour la détection d’anomalies dans des séries temporelles, en combinant :
				  collapsed:: true
					- la **capacité du Transformer** à modéliser les dépendances globales,
					- et la **compacité du CNN 1D** pour fusionner efficacement des représentations multi-niveaux.
					  
					  ---
				- **Contexte:** La détection d’anomalies dans des séries temporelles est cruciale pour la maintenance préventive dans de nombreux domaines (IT, industrie, santé). La rareté des anomalies rend les approches **non supervisées** attractives. Les approches classiques (LSTM, CNN, autoencodeurs, GAN) souffrent de limitations : dépendance locale, reconstruction imprécise, mauvaise gestion des variations globales.
				  
				  ---
				- **Principes clés:**
					- **Approche prédictive non supervisée** : le modèle est entraîné à prédire la suite d’une séquence normale.
					- Les anomalies sont détectées comme des points où **l’erreur de prédiction** est anormalement élevée.
					- Utilisation de **représentations multi-niveaux empilées** issues des couches intermédiaires du Transformer.
					  
					  ---
				- **Architecture du modèle:**
					- #### **1. Entrée**
					- Série temporelle univariée ou multivariée divisée en fenêtres glissantes de taille fixe `L`.
					- Chaque fenêtre devient une séquence d’entrée pour le modèle.
					- #### **2. Encodeur : Stacked Transformer Layers**
					- L’entrée est d’abord projetée dans un espace de dimension `d_model`.
					- Une **positional encoding** est ajoutée pour intégrer la notion de temps.
					- La séquence passe à travers `N` **couches Transformer encoders**, chacune dotée :
						- d’un **masked multi-head self-attention** (pas d’accès au futur),
						- et d’un feedforward network.
					- Les **représentations de chaque couche sont empilées** (`stack`) pour capturer :
						- les **dépendances locales** (couches basses),
						- les **dépendances globales** (couches hautes).
					- #### **3. Décodeur : 1D CNN**
					- La représentation empilée (shape : `d_model × N × L`) est fusionnée via une **convolution 1D** :
						- permet d'extraire efficacement les **motifs invariants** dans le temps.
						- compense les fluctuations locales ou le bruit.
					- Une couche **linéaire** prédit la suite de la séquence (horizon `τ`).
					- #### **4. Perte**
					- La fonction de perte est la **MSE** entre la séquence prédite et la séquence réelle décalée de `τ`.
					- Le modèle apprend à **prédire la dynamique normale** du signal.
					  
					  ---
				- **Score d’anomalie:**
					- Pour chaque point temporel, le score est :
					  
					  $s_t = \left\| x_t - \hat{x}_t \right\|_2$
					- Un **seuil dynamique** peut être défini (ex: `μ + kσ`) pour détecter les points comme anormaux.
					  
					  ---
				- **Avantages de STOC:**
					- **Fusion multi-niveaux** : exploite toute la richesse hiérarchique des couches Transformer.
					- **Convolution 1D efficace** : capture les régularités, atténue les anomalies isolées.
					- **Adaptable** à divers types de séries : bruitées, saisonnières, multivariées.
		- ---
		- ### Decompose Auto-Transformer Network (DATN)
		  link:: https://www.mdpi.com/2079-9292/12/2/354
		  title:: Decompose Auto-Transformer Time Series Anomaly Detection for Network Management
		  author:: Wu et al.
		  date:: 2023
		  journal:: Electronics, 12(2), 354
		  topics:: Outlier detection, Deep Learning, Transformers, time series
		  collapsed:: true
			- **Résumé rapide:** Ce modèle décompose les séries temporelles en composantes saisonnières et de tendance, puis utilise un Transformer pour modéliser ces composantes séparément. Cette approche permet de mieux capturer les motifs périodiques et les tendances à long terme pour une détection d'anomalies plus précise.
			- **Résumé complet:**
			  L’article présente **DATN (Decompose Auto-Transformer Network)**, un modèle d’apprentissage non supervisé pour la détection d’anomalies dans les séries temporelles, spécifiquement destiné à la gestion de réseaux. Le principal enjeu est de modéliser la complexité des dépendances temporelles et la nature stochastique des données réseaux. DATN s’appuie sur une **décomposition de séries temporelles** en composants **tendance** et **saisonnier**, couplée à des modules de **transformer auto-attentifs** pour améliorer la détection.
			  
			  ---
			- **Problem Formulation**
			  
			  Cette section formelle le problème comme suit :
			- Une **série temporelle multivariée** est représentée par $X = \{x_0, ..., x_{T-1}\}$, avec $x_t \in \mathbb{R}^m$.
			- L’objectif est de détecter les anomalies dans une série test $\hat{X}$ sans supervision, en produisant une séquence de sorties $Y = \{y_0, ..., y_{T-1}\}$, où $y_t \in \{0, 1\}$ indique la présence ou non d'une anomalie.
			  
			  La série est modélisée comme la **somme de deux composantes** :
			  
			  $$x_t = s_t + p_t$$
			- $s_t$​ : composante **saisonnière** (patterns périodiques)
			- $p_t$​ : composante **tendance** (évolution à long terme)
			  
			  L’architecture Transformer est décrite, avec attention multi-têtes et réseaux feedforward. Elle sert de base à la modélisation des relations temporelles dans les séries.
			  
			  ---
			- #### Section 4 – Decompose Auto-Transformer (DATN)
			  
			  Cette partie détaille l’architecture DATN, illustrée dans la Figure 1 de l’article.
				- **Décomposition:** Le bloc de décomposition sépare chaque série X en deux composantes :
				  $$Xt=MA(X), \quad Xs=X−Xt$$
				  **MA** est une moyenne mobile
				  $X_s$​ (saisonnier), $X_t​$ (tendance)
				- **Auto-Attention par FFT:** Une nouveauté clé est l’**auto-attention**, qui :
				  Utilise la **transformée de Fourier (FFT)** pour détecter les principales périodes dominantes dans la série.
				  Sélectionne les **top K fréquences** puis les reconvertit en domaine temporel (inverse FFT) pour renforcer les patterns périodiques dominants.
				  $$Γ_k=F(x_n), \quad x_n = \mathcal{F}^{-1}(\Gamma_k)$$
				  
				  Cette opération est appliquée **séparément aux composantes saisonnières et de tendance** avant leur passage dans les blocs Transformer.
				- **Encodage:** Chaque couche de l’encodeur suit cette séquence :
				  -Décomposition : $$X_s, X_t = \text{SeriesDecomp}(X)$$
				  -Auto-attention : extraction des patterns périodiques
				  -Attention multi-têtes
				  -Fusion additive des sorties $O_s + O_t$​
			- **Décodage et détection:** Un **simple décodeur linéaire** reconstruit la série.
			  L’anomalie est détectée par la **distance euclidienne** entre la série d'entrée et sa reconstruction :
			  $$s_t = \sum_{i=1}^m \| \hat{x}_t - x_t \|_2​$$
			  Un score élevé indique une anomalie probable.
			  
			  ---
			- #### Conclusion du modèle
			  
			  DATN améliore l’interprétation des séries complexes :
				- La **décomposition** simplifie les patterns.
				- L’**auto-attention FFT** renforce les périodicités significatives.
				- Le **décodage simplifié** favorise l’efficacité sans compromettre la performance.
		- ### Reversible Instance Normalized Anomaly Transformer
		  link:: https://www.mdpi.com/1424-8220/23/22/9272
		  title:: Anomaly Detection in Time Series Data Using Reversible Instance Normalized Anomaly Transformer
		  author:: Baidya & Jeong
		  date:: 2023
		  journal:: Sensors, 23(22), 9272
		  topics:: Outlier detection, Deep Learning, Transformers, time series
			- **Résumé rapide:** Ce modèle améliore l'architecture Anomaly Transformer en intégrant une normalisation d'instance réversible, ce qui permet de mieux gérer les variations de distribution dans les séries temporelles univariées et d'améliorer la détection d'anomalies.
			- **Résumé complet:**
			  collapsed:: true
				- **Objectif:** L'article propose **RINAT**, un modèle non supervisé pour la détection d'anomalies dans les séries temporelles, basé sur une version améliorée du **Anomaly Transformer**. Le modèle introduit deux innovations principales :
					- **Reversible Instance Normalization (RevIN)** appliquée uniquement aux associations de séries.
					- **Attention bi-branche** distinguant les associations **prior** (voisinage local) et **series** (global) pour mieux capturer les anomalies rares.
				- ---
				- **Méthode proposée : RINAT**
					- Anomaly Transformer (Rappel)
						- Basé sur l'architecture Transformer.
						- Introduit deux types d’attention :
							- **Series association** : attention classique entre tous les points temporels (par auto-attention).
							- **Prior association** : attention basée sur un **noyau gaussien** centré autour de chaque point.
						- La **discrépance d’association** (association discrepancy) est calculée via la **divergence KL** entre ces deux attentions.
					- ---
				- **Reversible Instance Normalization (RevIN):**
					- Normalise les séquences temporelles instance par instance (par spectre dans notre cas).
					- Le processus est réversible, permettant de **restaurer l’échelle d’origine**.
					- Elle est **appliquée uniquement sur les données servant au calcul des series associations**, car :
						- Les anomalies rares sont noyées lors de la normalisation.
						- Leur impact est donc préservé dans la branche *prior* qui reçoit les données non normalisées.
					- ---
					- **Architecture RINAT:**
						- #### Étapes clés :
							- **Embedding** : encodage linéaire des données temporelles.
							- **RevIN** : normalisation réversible pour la branche series.
							- **Attention duale** :
								- **Series attention** : via auto-attention sur les données normalisées.
								- **Prior attention** : via noyau gaussien learnable sur les données brutes.
							- **Association discrepancy** :
								- Calculée comme moyenne symétrique de KL(Series || Prior) et KL(Prior || Series).
							- **Feedforward + LayerNorm**
							- **Reconstruction + denormalization**
							- **Loss** :
								- Reconstruction loss (`||X - X̂||`)
								- Moins pondéré par la discrepancy (`- λ * discrepancy`), avec stratégie **minimax** :
									- *min* : prior s’adapte à series
									- *max* : series diverge du prior (pour renforcer les anomalies)
							- Fonction de perte finale : $\text{Loss} = ||X - \hat{X}|| - \lambda * KL(\text{Prior}, \text{Series})$
							- Score d’anomalie final : $AS(X) = \text{SoftMax}(-KL) × ||X - \hat{X}||$
- ## Stacking models applied to NIRS
  collapsed:: true
	- ### Pipeline stacking pour NIRS portable
	  link:: https://doi.org/10.1016/j.microc.2025.114056
	  title:: Application of portable near-infrared spectrometer based on stacking in the continuous counter-current extraction process of *Carthamus tinctorius* L.
	  author:: Zhang et al.
	  date:: 2025
	  journal:: Microchemical Journal, volume 214
	  topics:: NIRS, Stacking models
		- **Résumé rapide :** Utilisation de données NIRS, issues de spectromètres portables, sur la Carthame des teinturiers pour faire de la prédiction avec des modèles de stacking (PLSR, SVR, Lasso, RR, GPR // ELM) avec comme analyte la teneur en safflomine. Il y a d'abord une détection d'outliers et suppression (ACP -> intervalle de confiance à 95% + méthode boxplot). Plusieurs méthodes de pré-traitement sont comparées: SNV, mean center, autoscaling, First and second derivative, Savitzky-Golay, continuous wavelet transform, et les combinaisons de ces méthodes.
		  Plusieurs méthodes de sélection de variables sont appliquées et comparées: correlation coefficient, elimination of uninformative variables, variable importance in projection, competitive adaptative reweighted sampling pour le modèle de PLSR; random forest, maximal information coefficient, grey relation analysis, snake optimizer pour les 4 autres modèles de la stack.
	- ### Stacking with Ridge Regressors on multiple ensembles
		- link:: https://doi.org/10.1016/j.talanta.2024.126242
		  title:: Stacking and ridge regression-based spectral ensemble preprocessing method and its application in near-infrared spectral analysis
		  author:: Huang et al.
		  date:: 2024
		  journal:: Talanta, volume 276
		  topics:: NIRS, Stacking models, Ridge Regression
		- **Résumé rapide:** Les auteurs proposent de faire du stacking uniquement avec des RR (Ridge Regressor). 8 modèles RR sont dans la couche 1; 1 RR en tant que méta-learner. Les preprocessings utilisés sont l'identité, vector normalization, Savitzky-Golay, SG first derivative, SG second derivative, multiplicative scatter correction, standard normal variate, standard normal variate - detrending. Les performances sont meilleures pour ce modèle que pour les modèles individuels; sans pour autant vraiment augmenter le temps de calcul, puisqu'un modèle individuel passe aussi par une phase d'hyperparamétrisation complète, et que l'approche proposée ici est tout aussi efficace. Cette approche montre également de meilleurs résultats que plusieurs méthodes ensemblistes basées sur des pré-traitements multiples des données, souvent orientées PLS, à savoir SPORT, PORTO, MCCV stacked regression, et selective ensemble preprocessing PLS.
	- ### Stacking model with AWPSO on FT-NIR data
	  link:: https://doi.org/10.1016/j.infrared.2024.105169
	  title:: Multi-model fusion stacking ensemble learning method for the prediction of berberine by FT-NIR spectroscopy
	  author:: Li et al.
	  date:: 2024
	  journal:: Infrared Physics and Technology, volume 137
	  topics:: FT-NIR, Stacking models, Particle Swarm Optimization
		- **Résumé rapide:** Prédiction de la berbérine dans la plante chinoise Coptide (portée médicale et pharmaceutique). Utilisation d'un modèle de Stacking constitué de 3 modèles dans la première couche, sur des données FT-NIR. Plusieurs modèles de stacking ont été comparés en choisissant aléatoirement 3 modèles parmi les suivants: PLS, AdaBoost, GBDT, RF, XGBoost. Pour optimiser les paramètres de chaque modèle, la méthode *adaptative inertia weight particle swarm optimization (AWPSO)* est employée, après comparaison avec la méthode PSO classique. Les données sont splitées par méthode SPXY en 60/18 échantillons. Finalement, le meilleur modèle trouvé est constitué de RF, AdaBoost, XGBoost. Le méta-learner est la régression linéaire multiple. Le modèle de stacking a montré de meilleures performances que les modèles individuels optimisés.
	- ### Descriptive title
	  link:: https://doi.org/10.1016/j.still.2023.105684
	  title:: Predicting key soil properties from Vis-NIR spectra by applying dual-wavelength indices transformations and stacking machine learning approaches
	  author:: Tavakoli et al.
	  date:: 2023
	  journal:: Soil & Tillage Research, volume 229
	  topics:: NIRS, Stacking models, LUCAS dataset
		- **Résumé rapide :** Sur la base de données des sols LUCAS, plusieurs modèles sont comparés: PLS, PCR, deux modèles de stacking (PCR-poly, 2R-RF. Optimisation via grid search.
		  Le premier est basé sur une succession directe de transformations et de modèles linéaires: StandardScaler -> ACP (q>p variables résultantes) -> Régression Linéaire (vecteur de taille nx1 en sortie) -> Quantile Information Transformation -> Régression polynômiale. Optimisation via grid search.
		  Le deuxième entraîne un modèle de PLSR et un modèle Ridge en première couche, et un Random Forest en méta-learner. Optimisation via random search (sous-échantillonage aléatoire des combinaisons d'hyperparamètres essayés).
		  Les performances des modèles de stacking sont meilleures. Le meilleur modèle est PCR-poly.
		  Deux types de pré-traitements (NDI, SRI) sont appliqués aux spectres de réflectance bruts, ou lissés par Savitzky-Golay combinés à une moyennation par paquets de 20 longueurs d'onde. 
		  Pour deux bandes spectrales $R_i, R_j$: $$~~\text{NDI}_{ij} = \frac{R_j - R_i}{R_j + R_i}, ~~i<j$$
		  Et: $$~~\text{SRI}_{ij} = \frac{R_j}{R_i}, ~~i<j$$
		  
		  Les résultats sont meilleurs avec ces transformations NDI et SRI.
- ## Model Optimization
	- ### PLS models
		- ### Find the best number of components with MCCV
		  link:: https://www.sciencedirect.com/science/article/pii/S0169743900001222
		  title:: Monte Carlo cross validation
		  author:: Xu & Liang
		  date:: 2001
		  journal:: Chemometrics and Intelligent Laboratory Systems, Vol. 56, Issue 1, p1-11
		  topics:: Monte Carlo, cross validation, PLS, NIRS, ultraviolet spectroscopy
			- **Résumé rapide:** Plutôt que d'utiliser une méthode de validation croisée leave-one-out pour trouver le nombre optimal de composantes de la PLS, ce qui peut aboutir à de l'overfitting, il est pertinent d'appliquer une validation croisée par méthode de Monte Carlo. L'article le démontre en particulier sur de petites bases de données. 
			  Cette méthode désignée par l'acronyme MCCV consiste à séparer aléatoirement la base de données d'entraînement en deux pour la calibration et la validation. On répète $N$ fois le procédé pour calculer une métrique d'erreur, et on détermine le nombre de composantes à conserver comme celui qui minimise cette erreur. 
			  Les expériences effectuées dans l'article on utilise $N=2,5.n$, où $n$ est le nombre de spectres de la base de données. Ici 2 bases de données sont utilisées, avec $n\approx 50$. De plus, il est conclu que prendre une taille de validation entre 40 et 60% donne les meilleurs résultats. Il est même expliqué que pour de plus grandes bases de données, une taille de validation plus grande encore pourrait être plus intéressante. 
			  Pour la validation croisée leave-one-out, il arrive souvent que le nombre de composantes à garder soit surestimé. De même pour la méthode MCCV si la taille de validation n'est pas suffisamment grande.
- ## Template for articles
	- ### Descriptive title
	  link:: link to the see the article
	  title:: Article title
	  author:: Dupont et al.
	  date:: 2025
	  journal:: journal, conference, ...
	  topics:: main topics tackled
	  template:: article template