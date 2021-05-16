#set FLASK_APP=humanFlask.py
#set FLASK_ENV=development
#flask run --host=172.18.240.1


from keras.models import load_model
from flask import Flask,jsonify,request
from flask_restful import Resource,Api,reqparse
import base64
# import Image
from PIL import Image
import io
from keras.preprocessing.image import img_to_array
import numpy as np

import time
from keras.preprocessing.image import save_img

import  tensorflow as tf


app = Flask(__name__)
api=Api(app)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0
tf.config.experimental.set_memory_growth(physical_devices[0], True)



model=load_model('D:/study/python/AI/Deep Learning/Generative Adversarial Networks(GAN)/Pix2Pix/Human Generator/test.h5')

print("Model loaded")

def prepare_image(image,target):
    if image.mode != "RGB":
        image=image.convert("RGB")
    image=image.resize(target)
    image=img_to_array(image)

    image=(image - 127.5) / 127.5
    image=np.expand_dims(image,axis=0)
    return image



class Predict(Resource):
    def post(self):
        json_data=request.get_json()
        img_data=json_data['Image']

        # print(img_data);

        image=base64.b64decode(str(img_data))

        image=Image.open(io.BytesIO(image))

        # print(image)
         

        prepared_image=prepare_image(image,target=(256,256))

        # print(prepared_image)

        # print("Lets Predict")

        preds=model.predict(prepared_image)

        # print(preds,'This is preds')

        outputFile='output.png'
        savePath='./output/'

        outputs=tf.reshape(preds,[256,256,3])

        # print(outputs,'This is op')

        outputs=(outputs + 1) /2


        save_img(savePath+outputFile,img_to_array(outputs))

        # print("Saved")

        imageNew=Image.open(savePath+outputFile)
        imageNew=imageNew.resize((50,50))

        imageNew.save(savePath+"new_"+outputFile)

        with open(savePath+"new_"+outputFile,'rb') as image_file:
            encode_string=base64.b64encode(image_file.read())
        
        outputData={
            'Image':str(encode_string),

        }
        
        return outputData
        # return 1

    
api.add_resource(Predict,'/predict')

if __name__ == '__main__':
    app.run(debug=True)