import os
from tensorflow.python.keras import initializers, models
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.ops.gen_array_ops import shape
from tensorflow.python.ops.variables import trainable_variables
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers
from tensorflow.keras.datasets import mnist

physical_divice=tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_divice[0],True)


(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train.reshape(-1,28,28,1).astype("float32")/255.0
x_test=x_test.reshape(-1,28,28,1).astype("float32")/255.0


class Dense(layers.Layer):
    # def __init__(self,units,input_dim):
    #     super(Dense,self).__init__()
    #     self.w=self.add_weight(
    #         name='w',
    #         shape=(input_dim,units),
    #         initializer='random_normal',
    #         trainable=True,
    #     )
    def __init__(self,units):
        super(Dense,self).__init__()
        self.units=units
        # self.w=self.add_weight(
        #     name='w',
        #     shape=(input_dim,units),
        #     initializer='random_normal',
        #     trainable=True,
        # )

        # self.b=self.add_weight(
        #     name='b',
        #     shape=(units,),
        #     initializer='zeros',
        #     trainable=True,
        # )

        def build(self,input_shape):
            self.w=self.add_weight(
                name='w',
                shape=(input_shape[-1],self.units),
                initializer='random_normal',
                trainable=True,
            )

            self.b=self.add_weight(
                name='b',
                shape=(self.units,),
                initializer='zeros',
                trainable=True,
            )

        def call(self,inputs):
            return tf.matmul(inputs,self.w)+self.b
            

class MyRelu(layers.Layer):
    def __init__(self):
        super(MyRelu,self).__init__()
    
    def call(self, x):
        return tf.math.maximum(x,0)






class MyModel(keras.Model):
    def __init__(self,num_classes=10):
        super(MyModel,self).__init__()

        self.dense1=layers.Dense(64)
        self.dense2=layers.Dense(10)
        self.relu=MyRelu()

        # self.dense1=layers.Dense(64,784)
        # self.dense2=layers.Dense(10,64)


        # self.dense1=layers.Dense(64)
        # self.dense2=layers.Dense(num_classes)
    def call(self, input_tensor):
        x=self.relu(self.dense1(input_tensor))

        return self.dense2(x)


model=MyModel()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)
# 
print(y_train.shape,x_train.shape)

model.fit(x_train,y_train,epochs=2,verbose=2)
# model.evaluate(x_test,y_test,batch_size=32,verbose=2)

# model.save('pretrain2')