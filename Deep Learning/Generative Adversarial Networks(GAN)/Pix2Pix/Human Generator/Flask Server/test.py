import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


t1 = tf.Variable(["A"], ["B"], ["C"],tf.string)
# t2 = tf.Variable([["A","B"], ["C","D"]])
print(tf.math.angle(t1),'t2' ) 
# print(tf.rank(t1),'t1' ) # 3