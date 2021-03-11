WORKSPACE_PATH='Tensorflow/workspace'
SCRIPTS_PATH='Tensorflow/scripts'
APIMODEL_PATH='Tensorflow/models'
ANNOTATION_PATH=WORKSPACE_PATH+'/annotations'


IMAGE_PATH=WORKSPACE_PATH+'/images'
MODEL_PATH=WORKSPACE_PATH+'/models'

PRETRAINED_MODEL_PATH=WORKSPACE_PATH+'/pre-trained-models'

CONFIG_PATH=MODEL_PATH+'/my_ssd_mobnet/pipeline.config'

CHECKPOINT_PATH=MODEL_PATH+'/my_ssd_mobnet/'

labels=[
    {'name':'Mask','id':1},
    {'name':'NoMask','id':2}
]

with open(ANNOTATION_PATH+'\label_map.pbtxt','w') as f:
    for label in labels:
        f.write('item{\n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')


# python Tensorflow/scripts/generate_tfrecord.py -x Tensorflow/workspace/images/train -l Tensorflow/workspace/annotations/label_map.pbtxt -o Tensorflow/workspace/annotations/train.record
 