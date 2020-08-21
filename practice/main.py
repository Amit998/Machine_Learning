import tensorflow as tf
# print(tf.version)
string = tf.Variable("this is a string", tf.string) 
number = tf.Variable(324, tf.int16)
floating = tf.Variable(1.567,tf.float64)
rank1_tensor = tf.Variable(["Test"], tf.string) 
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)
tf.shape(rank2_tensor)
tf.rank(rank1_tensor)