import tensorflow as tf
from tensorflow.python.autograph.impl.api import convert
from tensorflow.python.keras.saving.save import save_model

save_model_dir=""
convert=tf.lite.TFLiteConverter.from_saved_model(save_model_dir)

convert.optimizations=[tf.lite.Optimize.DEFAULT]
