import tensorflow as tf
tf.enable_eager_execution()

a = tf.sequence_mask(3, 5)
print(a)
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
b = (-2 ** 16 + 1)
print(b)
# paddings = tf.ones_like(tensor) * (-2 ** 32 + 1)
paddings = tf.ones_like(tensor)
print(paddings)
