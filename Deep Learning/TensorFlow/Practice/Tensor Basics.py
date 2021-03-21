import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
physical_divice=tf.config.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(physical_divice[0],True)
# print(physical_divice)

# print(tf.__version__)

# x=tf.constant(4,shape=(1,1),dtype=tf.float32)
# x=tf.constant([[1,2,3],[4,5,6]])
# x=tf.ones((3,3))
# x=tf.zeros((3,3))
# x=tf.eye(2)
# x=tf.random.normal((3,3),mean=0,stddev=1)

# x=tf.random.uniform((1,3),minval=0,maxval=1)
# x=tf.range(start=1,limit=10,delta=2)
# x=tf.cast(x,dtype=tf.float64)


# # print(x)


# #math

# x=tf.constant([1,2,3])
# y=tf.constant([9,8,7])

# # z=tf.add(x,y)
# # z1=x+y
# # print(z1)

# # z=tf.subtract(x,y)
# # z1=x-y
# # print(z1)

# # z=tf.divide(x,y)
# # z1=x/y
# # print(z1)


# # z=tf.multiply(x,y)
# # z1=x*y
# # print(z)


# # z=tf.tensordot(x,y,axes=1)
# # z=tf.reduce_sum(x*y,axis=None)

# # z1=xy

# # z=x ** y
# # print(z)

# x=tf.random.normal((2,3))
# y=tf.random.normal((3,4))

# z=tf.matmul(x,y)

# print(z)
# z=x @ y
# print(z)

# x=tf.constant([0,1,2,3,4,5,6])
# print(x[:])


# indices=tf.constant([0,3])
# x_ind=tf.gather(x,indices)
# print(x_ind)


x=tf.constant([[0,3],
                [2,1],
                [4,2],
                [3,1]])
# print(x[0,:])
# print(x[0:2,:])

x=tf.range(9)
x=tf.reshape(x,(3,3))

print(x)
x=tf.transpose(x,perm=[1,0])

print(x)