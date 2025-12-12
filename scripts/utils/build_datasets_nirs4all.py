import os
import pandas as pd
from pathlib import Path

SRC_ROOT = Path("Data/Regression")
DST_ROOT = Path("Data_nirs4all/Regression")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def detect_sep(path):
    with open(path) as f:
        first = f.readline()
    return ";" if ";" in first else ","


def load_csv(path):
    sep = detect_sep(path)
    return pd.read_csv(path, sep=sep)


def convert_dataset(src_dir: Path, dst_dir: Path):
    print(f"🔧 Converting {src_dir.name} → {dst_dir}")

    ensure_dir(dst_dir)

    # Input CSVs
    Xcal = src_dir / "Xcal.csv"
    Ycal = src_dir / "Ycal.csv"
    Xval = src_dir / "Xval.csv"
    Yval = src_dir / "Yval.csv"

    for f in [Xcal, Ycal, Xval, Yval]:
        if not f.exists():
            raise FileNotFoundError(f"❌ Missing file: {f}")

    # Load data
    Xtrain = load_csv(Xcal)
    Ytrain = load_csv(Ycal)
    Xtest  = load_csv(Xval)
    Ytest  = load_csv(Yval)

    # Save exactly like regression_2
    Xtrain.to_csv(dst_dir / "Xtrain.csv", sep=";", index=False)
    Ytrain.to_csv(dst_dir / "Ytrain.csv", sep=";", index=False)
    Xtest.to_csv(dst_dir / "Xtest.csv", sep=";", index=False)
    Ytest.to_csv(dst_dir / "Ytest.csv", sep=";", index=False)

    # OPTIONAL metadata → create minimal Mtrain/Mtest
    Mtrain = pd.DataFrame({"ID": range(len(Xtrain))})
    Mtest = pd.DataFrame({"ID": range(len(Xtest))})

    Mtrain.to_csv(dst_dir / "Mtrain.csv", sep=";", index=False)
    Mtest.to_csv(dst_dir / "Mtest.csv", sep=";", index=False)

    print(f"✅ Dataset created in {dst_dir}")


def main():
    ensure_dir(DST_ROOT)

    for ds in sorted(SRC_ROOT.iterdir()):
        if ds.is_dir():
            dst = DST_ROOT / ds.name
            convert_dataset(ds, dst)

    print("✨ All datasets converted successfully.")


if __name__ == "__main__":
    main()
