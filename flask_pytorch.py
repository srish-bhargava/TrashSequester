from flask import Flask, jsonify, request
from PIL import Image
from torchvision import models
from torch.autograd import Variable
import torchvision.transforms as transforms
from tensorflow import keras
import io
import os
import cv2
import numpy as np
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = keras.models.load_model("model.h5")
labels=['O','R']

def image_process(image):
    image=cv2.imread('DATASET/TEST/R/R_10018.jpg')
    image=cv2.resize(image,(150,150))
    image=np.array(image,dtype=np.float32)
    image = np.array(image)
    image = image / 255
    image = np.array([image])
    return image

@app.route('/', methods = ['POST'])
def model_evaluate():
    data = {"success":False}
    if request.method == "POST":
    
        if request.files.get("image"):
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = image_process(image)
            output = model(image)
            print (output)
            classes = np.argmax(output, axis = 1)
            if classes==0:
                res="Organic"
            else:
                res= "Recyclable"
            data["predictions"] = "The trash is " + str(res)
            data["success"] = True

    return jsonify(data)

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8001)