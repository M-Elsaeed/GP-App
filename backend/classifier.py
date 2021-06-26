import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIL import ImageFile, Image
from numpy import expand_dims
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from cv2 import cv2
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = load_model("./model.h5") #ResNet50(weights='imagenet')

def getPrediction(img_bytes, model):
    #read image file string data
    filestr = img_bytes.read()
    #convert string data to numpy array
    numpy_image = np.fromstring(filestr, np.uint8)
    # convert np array to image
    numpy_image = cv2.imdecode(numpy_image, cv2.IMREAD_COLOR)    
    # print("Initial Shape", numpy_image.shape)
    
    numpy_image = cv2.resize(numpy_image, (224, 224))
    # print("After Resize", numpy_image.shape)
    # print(numpy_image[0])
    numpy_image = numpy_image.astype('float32')
    numpy_image/=255
    image_batch = expand_dims(numpy_image, axis=0)
    # print("After Expansion", image_batch.shape)
    # print(image_batch[0][0])

    preds = model.predict(image_batch)
    
    return preds

def classifyImage(file):
    # Returns a probability scores matrix 
    preds = getPrediction(file, model)[0]
    temporalResult = None
    if preds[0] > preds[1] and preds[0] > preds[2]:
        temporalResult = "L"
    elif preds[1] > preds[0] and preds[1] > preds[2]:
        temporalResult = "R"
    else:
        temporalResult = "F"
    return temporalResult