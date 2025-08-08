import numpy as np

def get_pls_component_candidates(n_spectra, n_wavelengths, prior_components, max_evals, cv, rd_seed=42):
    rng = np.random.default_rng(rd_seed)

    nb_spectra_cv = int(n_spectra * (cv - 1) / cv)
    max_comp = min(n_wavelengths, nb_spectra_cv)

    if len(prior_components) == 0:
        # No prior knowledge → uniform search
        return np.linspace(1, max_comp, max_evals, dtype=int)
    
    # Compute median and std from prior results
    mu = int(np.median(prior_components))
    sigma_empirical = np.std(prior_components)
    
    # Prevent sigma from being too small
    sigma_min = max(2, max_comp * 0.05)  # 5% of wavelengths or at least 2 comps
    sigma = max(sigma_empirical, sigma_min)
    
    # Generate candidates from truncated normal distribution
    candidates = []
    while len(candidates) < max_evals:
        # Draw a batch from normal
        batch = rng.normal(loc=mu, scale=sigma, size=max_evals*2)
        batch = np.round(batch).astype(int)
        
        # Keep only within valid range
        batch = batch[(batch >= 1) & (batch <= max_comp)]
        
        # Append unique ones until reaching max_evals
        for b in batch:
            if b not in candidates:
                candidates.append(b)
                if len(candidates) >= max_evals:
                    break
    
    return np.array(candidates, dtype=int)