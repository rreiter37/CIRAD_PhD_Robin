import os
import shutil
import pandas as pd

ROOT_SRC = "Data/Regression"
ROOT_DST = "Data_nirs4all/Regression"


def ensure_dir(path):
    """Create folder if missing."""
    os.makedirs(path, exist_ok=True)


def convert_dataset(dataset_name):
    """
    Convert a dataset folder:
        Data/Regression/{dataset_name}/Xcal.csv → Data_nirs4all/.../train/X.csv
        Data/Regression/{dataset_name}/Ycal.csv → ... train/Y.csv
        Data/Regression/{dataset_name}/Xval.csv → ... test/X.csv
        Data/Regression/{dataset_name}/Yval.csv → ... test/Y.csv
    """

    src_dir = os.path.join(ROOT_SRC, dataset_name)
    dst_dir = os.path.join(ROOT_DST, dataset_name)

    # Detect required files
    Xcal = os.path.join(src_dir, "Xcal.csv")
    Ycal = os.path.join(src_dir, "Ycal.csv")
    Xval = os.path.join(src_dir, "Xval.csv")
    Yval = os.path.join(src_dir, "Yval.csv")

    required = [Xcal, Ycal, Xval, Yval]
    missing = [f for f in required if not os.path.exists(f)]

    if missing:
        print(f"❌ Dataset {dataset_name} skipped; missing files:")
        for m in missing:
            print("   -", m)
        return

    # Prepare output structure
    train_dir = os.path.join(dst_dir, "train")
    test_dir  = os.path.join(dst_dir, "test")

    ensure_dir(train_dir)
    ensure_dir(test_dir)

    # Copy files with correct names
    shutil.copy(Xcal, os.path.join(train_dir, "X.csv"))
    shutil.copy(Ycal, os.path.join(train_dir, "Y.csv"))
    shutil.copy(Xval, os.path.join(test_dir,  "X.csv"))
    shutil.copy(Yval, os.path.join(test_dir,  "Y.csv"))

    print(f"✅ Converted dataset: {dataset_name}")


def main():
    """Scan all datasets inside Data/Regression and convert them."""
    if not os.path.exists(ROOT_SRC):
        print("❌ Source folder not found:", ROOT_SRC)
        return

    datasets = sorted(os.listdir(ROOT_SRC))
    print(f"🔍 Found {len(datasets)} dataset folders in {ROOT_SRC}/")

    for name in datasets:
        folder = os.path.join(ROOT_SRC, name)
        if os.path.isdir(folder):
            convert_dataset(name)

    print("\n✨ Conversion complete.")
    print(f"New datasets available inside {ROOT_DST}/")


if __name__ == "__main__":
    main()
