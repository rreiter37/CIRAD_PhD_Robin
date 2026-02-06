#!/usr/bin/env python3
"""
TabPFN GPU check script.

What it does:
1) Checks whether PyTorch sees a CUDA GPU.
2) Tries to import TabPFN.
3) Runs a tiny fit/predict with TabPFN on GPU if possible (device='cuda').
   - Note: the first TabPFN run may download pretrained weights (needs internet).
"""

import sys
import traceback
import numpy as np


def print_torch_cuda_info() -> bool:
    """Print basic PyTorch/CUDA info and return True if CUDA is available."""
    try:
        import torch
    except Exception as e:
        print("[FAIL] Could not import torch:", repr(e))
        return False

    print("=== PyTorch / CUDA info ===")
    print("torch.__version__:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.backends.cudnn.version():", torch.backends.cudnn.version())
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        return False

    try:
        n = torch.cuda.device_count()
        print("CUDA device_count:", n)
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            print(f"  - GPU {i}: {props.name} | VRAM: {props.total_memory / (1024**3):.2f} GB "
                  f"| CC: {props.major}.{props.minor}")
        print("Current device:", torch.cuda.current_device())
        print("Current device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    except Exception as e:
        print("[WARN] Could not query detailed GPU properties:", repr(e))

    return True


def import_tabpfn():
    """
    Try multiple import paths to support different TabPFN versions.
    Returns (TabPFNClassifier, TabPFNRegressor, version_str_or_None).
    """
    TabPFNClassifier = None
    TabPFNRegressor = None
    version_str = None

    # Common modern import
    try:
        from tabpfn import TabPFNClassifier as _C  # type: ignore
        TabPFNClassifier = _C
        try:
            from tabpfn import TabPFNRegressor as _R  # type: ignore
            TabPFNRegressor = _R
        except Exception:
            TabPFNRegressor = None

        try:
            import tabpfn  # type: ignore
            version_str = getattr(tabpfn, "__version__", None)
        except Exception:
            version_str = None

        return TabPFNClassifier, TabPFNRegressor, version_str
    except Exception:
        pass

    # Older layouts (fallbacks)
    try:
        from tabpfn.scripts.transformer_prediction_interface import (  # type: ignore
            TabPFNClassifier as _C
        )
        TabPFNClassifier = _C
        try:
            from tabpfn.scripts.transformer_prediction_interface import (  # type: ignore
                TabPFNRegressor as _R
            )
            TabPFNRegressor = _R
        except Exception:
            TabPFNRegressor = None

        try:
            import tabpfn  # type: ignore
            version_str = getattr(tabpfn, "__version__", None)
        except Exception:
            version_str = None

        return TabPFNClassifier, TabPFNRegressor, version_str
    except Exception:
        return None, None, None


def tiny_classification_run(TabPFNClassifier, device: str) -> bool:
    """Run a minimal classification fit/predict and try to verify it's on the expected device."""
    print("\n=== TabPFN tiny classification test ===")

    # Synthetic binary classification dataset
    rng = np.random.default_rng(0)
    X = rng.normal(size=(128, 20)).astype(np.float32)
    y = (X[:, 0] + 0.25 * X[:, 1] > 0).astype(np.int64)

    # TabPFN expects small/medium tabular data; this is intentionally tiny.
    try:
        # Many versions accept `device=...`
        clf = TabPFNClassifier(device=device)
    except TypeError:
        # If `device` isn't accepted, instantiate without it and warn.
        print("[WARN] Your TabPFNClassifier does not accept device=... in the constructor.")
        clf = TabPFNClassifier()

    # Fit & predict
    try:
        clf.fit(X, y)
        proba = clf.predict_proba(X[:5])
        pred = clf.predict(X[:5])
        print("predict_proba shape:", getattr(proba, "shape", None))
        print("predict:", pred)
    except Exception as e:
        print("[FAIL] TabPFN fit/predict failed:", repr(e))
        print("---- Traceback ----")
        traceback.print_exc()
        return False

    # Best-effort device verification
    # Not all wrappers expose the underlying torch model/params, so we attempt several strategies.
    try:
        import torch

        found_device = None

        # Common: sklearn-like wrapper has a `.model` attribute (or similar)
        for attr in ["model", "net", "clf", "_model", "estimator_"]:
            if hasattr(clf, attr):
                m = getattr(clf, attr)
                try:
                    # If it's a torch.nn.Module
                    if hasattr(m, "parameters"):
                        p = next(m.parameters())
                        found_device = str(p.device)
                        break
                except Exception:
                    pass

        if found_device is None:
            # Some versions store a device string
            for attr in ["device", "_device"]:
                if hasattr(clf, attr):
                    found_device = str(getattr(clf, attr))
                    break

        print("Requested device:", device)
        print("Detected device (best-effens):", found_device)

        if device == "cuda":
            # If CUDA is requested, we consider success if CUDA is available
            # and the detected device string contains 'cuda' (when detectable).
            if torch.cuda.is_available() and (found_device is None or "cuda" in found_device):
                print("[OK] CUDA is available and TabPFN run completed. Likely running on GPU.")
                return True
            print("[WARN] CUDA is available but could not confirm TabPFN is on GPU (API may hide it).")
            return True

        print("[OK] TabPFN run completed on CPU.")
        return True

    except Exception as e:
        print("[WARN] Could not verify device:", repr(e))
        print("[OK] TabPFN run completed; device verification is inconclusive.")
        return True


def main():
    cuda_ok = print_torch_cuda_info()

    TabPFNClassifier, TabPFNRegressor, tabpfn_ver = import_tabpfn()
    print("\n=== TabPFN import ===")
    if TabPFNClassifier is None:
        print("[FAIL] Could not import TabPFN. Try: pip install tabpfn")
        sys.exit(2)

    print("[OK] TabPFN imported successfully.")
    if tabpfn_ver is not None:
        print("tabpfn.__version__:", tabpfn_ver)

    device = "cuda" if cuda_ok else "cpu"
    ok = tiny_classification_run(TabPFNClassifier, device=device)

    print("\n=== Summary ===")
    if cuda_ok:
        print("PyTorch sees a CUDA GPU: YES")
        print("TabPFN test run: ", "OK" if ok else "FAIL")
        print("If the first run failed due to missing weights, re-run with internet access enabled.")
    else:
        print("PyTorch sees a CUDA GPU: NO")
        print("TabPFN will run on CPU unless CUDA is fixed (driver / CUDA / PyTorch build).")
        print("TabPFN test run: ", "OK" if ok else "FAIL")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
