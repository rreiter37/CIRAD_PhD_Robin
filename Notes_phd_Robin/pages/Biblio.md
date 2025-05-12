## Detection of outliers [[Outlier Detection]]
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
			- **Method 2 :**  A first approach based on **Data Depth Theory**. Let A be defined as the matrix of absorbances of size $N_T \times n_x$, containing $N_T$ recorded spectra. Each spectrum measures $n_x$ wavelenghts.
			  
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
		-
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
			- ### Collective outlier detection with LSTM
			  link:: https://www.researchgate.net/publication/309370951_Collective_Anomaly_Detection_Based_on_Long_Short-Term_Memory_Recurrent_Neural_Networks
			  title:: Collective Anomaly Detection Based on Long Short-Term Memory Recurrent Neural Networks
			  author:: Bontemps et al.
			  date:: 2016
			  journal:: Future Data and Security Engineering (FDSE conference)
			  topics:: Outlier detection, Deep Learning, LSTM, RNN, time series
			- ### Outlier detection with Bi-LSTM
			  link:: https://www.researchgate.net/publication/369368513_A_Bi-LSTM_Autoencoder_Framework_for_Anomaly_Detection_--_A_Case_Study_of_a_Wind_Power_Dataset
			  title:: A Bi-LSTM Autoencoder Framework for Anomaly Detection -- A Case Study of a Wind Power Dataset
			  author:: Raihan & Ahmed
			  date:: 2023
			  journal:: IEEE 19th Conference on Automation and Engineering (CASE)
			  topics:: Outlier detection, Deep Learning, LSTM, RNN, time series
		- #### HTM
		  collapsed:: true
			- ### Real-time anomaly detection with HTM
			  link:: https://www.sciencedirect.com/science/article/pii/S0925231217309864
			  title:: Unsupervised real-time anomaly detection for streaming data
			  author:: Ahmad et al.
			  date:: 2017
			  journal:: Neurocomputing (volume 262, pages 134-147)
			  topics:: Outlier detection, Deep Learning, HTM, time series
			- ### Anomaly detection with HTM
			  link:: https://www.sciencedirect.com/science/article/pii/S0925231217313887
			  title:: Hierarchical Temporal Memory method for time-series-based anomaly detection
			  author:: Wu et al.
			  date:: 2018
			  journal:: Neurocomputing (volume 273, pages 535-546)
			  topics:: Outlier detection, Deep Learning, HTM, time series
		- #### CNN
		  collapsed:: true
			- ### Anomaly detection with CNN
			  link:: https://arxiv.org/pdf/1906.03821
			  title:: Time-Series Anomaly Detection Service at Microsoft
			  author:: Ren et al.
			  date:: 2019
			  journal:: Association for Computing Machinery
			  topics:: Outlier detection, Deep Learning, CNN, time series
		- #### VAE
		  collapsed:: true
			- ### Anomaly detection with VAE for Seasonal KPIs
			  link:: https://dl.acm.org/doi/pdf/10.1145/3178876.3185996
			  title:: Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications
			  author:: Xu et al.
			  date:: 2018
			  journal:: International World Wide Web Conferences Steering Committee
			  topics:: Outlier detection, Deep Learning, VAE, time series
			- ### Anomaly Detection with conditional variational autoencodeur
			  link:: https://www.researchgate.net/publication/333072648_Robust_and_Unsupervised_KPI_Anomaly_Detection_Based_on_Conditional_Variational_Autoencoder
			  title:: Robust and Unsupervised KPI Anomaly Detection Based on Conditional Variational Autoencoder
			  author:: Li et al.
			  date:: 2018
			  journal:: IPCC Conference
			  topics:: Outlier detection, Deep Learning, VAE, time series
			- ### Anomaly Detection with Adversarial Training of VAE
			  link:: https://www.researchgate.net/publication/333851045_Unsupervised_Anomaly_Detection_for_Intricate_KPIs_via_Adversarial_Training_of_VAE
			  title:: Unsupervised Anomaly Detection for Intricate KPIs via Adversarial Training of VAE
			  author:: Chen et al.
			  date:: 2019
			  journal:: IEEE Conference on Computer Communications INFOCOM
			  topics:: Outlier detection, Deep Learning, VAE, time series
		- #### AE
		  collapsed:: true
			- ### Anomaly detection with AE
			  link:: https://www.researchgate.net/publication/304758073_LSTM-based_Encoder-Decoder_for_Multi-sensor_Anomaly_Detection
			  title:: LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection
			  author:: Malhotra et al.
			  date:: 2016
			  journal:: 
			  topics:: Outlier detection, Deep Learning, AE, time series
- ### Using the architecture of transformers
	- #### Founding article on the architecture of transformers
	  link:: https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
	  title:: Attention Is All You Need
	  author:: Vaswani et al.
	  date:: 2017
	  journal:: Computing Research Repository (CoRR)
	  topics:: Outlier detection, Deep Learning, Transformers, attention mechanisms
	  collapsed:: true
		- This paper presents the architecture of Transformers for transduction tasks. It consists of using Multi-Head Attention layers in both encoder and decoder processes. It allows a less sequential structure, and subsequently the possibility to parallelize tasks during the training phase, thus it reduces the computation time compared to sequential and convolutional layers.
- ## Template for articles
	- ### Descriptive title
	  link:: link to the see the article
	  title:: Article title
	  author:: Dupont et al.
	  date:: 2025
	  journal:: journal, conference, ...
	  topics:: main topics tackled
	  template:: article template