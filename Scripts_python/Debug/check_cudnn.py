import os
import glob
import subprocess

print("🔍 Vérification des fichiers cuDNN...")

cuda_path = "/usr/local/cuda"
lib_path = os.path.join(cuda_path, "lib64")
include_path = os.path.join(cuda_path, "include")

# 1. Fichiers .so et .h nécessaires
lib_files = glob.glob(os.path.join(lib_path, "libcudnn*"))
include_files = glob.glob(os.path.join(include_path, "cudnn*.h"))

print("\n📁 Fichiers trouvés dans lib64:")
if lib_files:
    for f in lib_files:
        print("  ✅", f)
else:
    print("  ❌ Aucun fichier libcudnn*.so trouvé")

print("\n📁 Fichiers trouvés dans include:")
if include_files:
    for f in include_files:
        print("  ✅", f)
else:
    print("  ❌ Aucun fichier cudnn*.h trouvé")

# 2. Vérifier la version de libcudnn si disponible
libcudnn_so = next((f for f in lib_files if "libcudnn.so" in f), None)
if libcudnn_so:
    try:
        output = subprocess.check_output(["strings", libcudnn_so])
        version_lines = [line for line in output.decode().splitlines() if "cudnn" in line.lower()]
        print("\n🔢 Version probable de cuDNN trouvée dans libcudnn.so :")
        for line in version_lines:
            print("  🔹", line)
    except Exception as e:
        print("⚠️ Impossible d’extraire la version :", e)

# 3. LD_LIBRARY_PATH
ld_path = os.environ.get("LD_LIBRARY_PATH", "")
print("\n📌 LD_LIBRARY_PATH contient :")
if "/usr/local/cuda/lib64" in ld_path:
    print("  ✅ /usr/local/cuda/lib64 est bien dans LD_LIBRARY_PATH")
else:
    print("  ❌ /usr/local/cuda/lib64 n'est pas dans LD_LIBRARY_PATH")

# Résumé
if lib_files and include_files and "/usr/local/cuda/lib64" in ld_path:
    print("\n✅ cuDNN semble correctement installé.")
else:
    print("\n⚠️ Il manque des composants ou la variable d'environnement est incorrecte.")
