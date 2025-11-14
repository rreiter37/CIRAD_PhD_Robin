import tensorflow as tf
import torch
import time

print("=== TensorFlow ===")
print("TF version:", tf.__version__)
print("GPUs disponibles :", tf.config.list_physical_devices('GPU'))

try:
    start = time.time()
    a = tf.random.normal([10000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.linalg.matmul(a, b)
    _ = c.numpy()  # forcer exécution
    end = time.time()
    print("TensorFlow matmul GPU OK, temps:", end - start, "sec")
except Exception as e:
    print("Erreur TensorFlow :", e)

print("\n=== PyTorch ===")
print("Torch version:", torch.__version__)
print("CUDA disponible :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))
try:
    start = time.time()
    a = torch.randn(10000, 1000).cuda()
    b = torch.randn(1000, 1000).cuda()
    c = torch.matmul(a, b)
    _ = c.cpu().numpy()  # ramener CPU pour synchroniser
    end = time.time()
    print("PyTorch matmul GPU OK, temps:", end - start, "sec")
except Exception as e:
    print("Erreur PyTorch :", e)

