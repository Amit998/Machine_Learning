# from tensorflow.python.client import device_lib
# import tensorflow as tf

# # print(device_lib.list_local_devices())
# print(tf.test.gpu_device_name())

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)


# python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --num_train_steps=5000