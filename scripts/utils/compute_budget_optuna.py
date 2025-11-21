def compute_budget_optuna(n_samples):
    """Compute adaptive Optuna budget based on dataset size."""

    if n_samples < 200:
        return {
            "n_trials_initial": 60,
            "n_trials_refine": 10,
        }
    elif n_samples < 2000:
        return {
            "n_trials_initial": 200,
            "n_trials_refine": 30,
        }
    elif n_samples < 8000:
        return {
            "n_trials_initial": 350,
            "n_trials_refine": 60,
        }
    else:
        return {
            "n_trials_initial": 500,
            "n_trials_refine": 200,
        }