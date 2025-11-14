# --------------------------------------------------------------
# Function to build consistent filenames for results, figures, and timings
def build_filename(prefix, data_source, top_n=None, epochs=None, patience=None,
                   optim_type="uniform", model_names=None, adaptive_batch_size=False,
                   extension="csv", replace_str=None):
    """
    Build a standardized filename with optional suffixes.
    - prefix: base name like "results", "heatmap", "timing_results", etc.
    - data_source: dataset name
    - top_n: if not None, include top N preprocessings
    - epochs/patience: included only for CNN models
    - optim_type: "uniform" or "progressive"
    - model_names: list of model names or None
    - adaptive_batch_size: add "_static_batch_size" if True or "static"; "_dynamic_batch_size" if "dynamic"
    - extension: file extension (e.g., "csv", "png")
    - replace_str: if not None, replace "results" in the name with another string
    """
    # Start with the prefix and dataset
    name_parts = [prefix, data_source]

    # Add top_n if provided
    if top_n is not None:
        name_parts.append(f"top_{top_n}")

    # Add epochs/patience only if CNN model is involved
    if model_names is not None and "CNN" in "_".join(model_names):
        if epochs is not None:
            name_parts.append(f"{epochs}epochs")
        if patience is not None:
            name_parts.append(f"{patience}patience")

    # Add optimization type
    name_parts.append(f"{optim_type}_optim")

    # Add model names if provided
    if model_names is not None:
        name_parts.append("_".join(model_names))

    # Add adaptive batch size if activated
    if adaptive_batch_size == "static":
        name_parts.append("static_batch_size")
    elif adaptive_batch_size == "dynamic":
        name_parts.append("dynamic_batch_size")

    # Join parts with underscores
    filename = "_".join(map(str, name_parts))

    # Replace "results" if needed (for F1, FPR, per_model, etc.)
    if replace_str is not None:
        filename = filename.replace("results", replace_str)

    return f"{filename}.{extension}"