import os
import tarfile
from pathlib import Path

def compress_pth_files(root_dir, delete_original=False):
    root = Path(root_dir)
    pth_files = list(root.rglob("*.pth"))

    print(f"🔍 {len(pth_files)} found .pth files to compress.")

    for pth_path in pth_files:
        tar_path = pth_path.with_suffix(".pth.tar.gz")

        print(f"Compression of : {pth_path}")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(pth_path, arcname=pth_path.name)

        if delete_original:
            print(f"Delete the original file : {pth_path}")
            pth_path.unlink()

    print("Compression done.")


compress_pth_files("Outputs/outliers_detection/Fractal_noise/Models", delete_original=True)
