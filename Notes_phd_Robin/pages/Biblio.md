## Template
	- ### 
	  link:: 
	  title:: 
	  author:: 
	  date:: 
	  journal:: 
	  topics:: 
	  template:: article template
- ## Detection of outliers [[Outlier Detection]]
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
	- ### Outlier Detection with deep learning techniques on time series
	  link:: https://dl.acm.org/doi/10.1145/3691338
	  title:: Deep Learning for Time Series Anomaly Detection: A Survey 
	  author:: Darban et al.
	  date:: 2024
	  journal:: ACM Computing Surveys, Volume 57, Issue 1
	  topics:: Outlier detection, Deep Learning, time series