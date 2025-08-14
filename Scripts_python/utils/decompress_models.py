import tarfile

with tarfile.open("model.pth.tar.gz", "r:gz") as tar:
    tar.extractall(path=".")
    print("Model extracted successfully.")