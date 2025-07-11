import tensorflow as tf
import time

with tf.device('/CPU:0'):
    start = time.time()
    tf.linalg.matmul(tf.random.normal([10000, 1000]), tf.random.normal([1000, 1000]))
    print("CPU:", time.time() - start)

with tf.device('/GPU:0'):
    start = time.time()
    tf.linalg.matmul(tf.random.normal([10000, 1000]), tf.random.normal([1000, 1000]))
    print("GPU:", time.time() - start)


