import numpy as np

def get_pls_component_candidates(n_spectra, n_wavelengths, prior_components, max_evals, cv, rd_seed=42):
    rng = np.random.default_rng(rd_seed)

    nb_spectra_cv = int(n_spectra * (cv - 1) / cv)
    max_comp = min(n_wavelengths, nb_spectra_cv)

    # Security: we can't have more unique candidates than possible components
    max_evals = min(max_evals, max_comp)

    if len(prior_components) == 0:
        return np.linspace(1, max_comp, max_evals, dtype=int)
    
    mu = int(np.median(prior_components))
    sigma_empirical = np.std(prior_components)
    sigma_min = max(2, max_comp * 0.05)
    sigma = max(sigma_empirical, sigma_min)
    
    candidates = set()
    attempts = 0
    max_attempts = 1000  # safety stop

    while len(candidates) < max_evals and attempts < max_attempts:
        batch = rng.normal(loc=mu, scale=sigma, size=max_evals*2)
        batch = np.round(batch).astype(int)
        batch = batch[(batch >= 1) & (batch <= max_comp)]
        candidates.update(batch)
        attempts += 1

    return np.array(list(candidates)[:max_evals], dtype=int)