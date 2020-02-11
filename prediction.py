from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import load_model
from keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import colors
import keras.backend as K
import numpy as np
import cv2
import sys
from skimage import color
from skimage.morphology import disk
from skimage import img_as_ubyte
from skimage.filters.rank import enhance_contrast_percentile
from skimage import io
from PIL import Image, ImageFilter,ImageEnhance
from tensorflow.keras.models import model_from_json
from datetime import datetime

def load_img(imag_path):
    img_path = imag_path
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x /= 255.
    x = np.expand_dims(x, axis=0)
    return x

def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


def grad_cam(model,img,lab_pred,layer):
    mod_out = model.output[0, lab_pred]
    conv_output = model.get_layer(layer).output
    grads = K.gradients(mod_out, conv_output)[0]
    grads = normalize(grads)
    gradient_function = K.function([model.input], [conv_output, grads])
    output, grads_val = gradient_function([img])
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam_max = cam.max() 
    if cam_max != 0:
        cam = cam / cam_max
    return cam

def ind_pred(img,model):
    preds = model.predict(img)
    ind = np.argmax(preds)
    return ind


def load_image(path, preprocess=True):
    """Load image for plot"""
    x = image.load_img(path, target_size=(H, W))
    return x


def plot_oct(cam,img_path,etiqueta):
    plt.figure(figsize=(15, 10))
    plt.title(etiqueta)
    plt.axis('off')
    im = Image.open(img_path)
    im.thumbnail((299, 299), Image.ANTIALIAS)
    im.save("img2_temp.jpeg")    
    plt.imshow(load_image("img2_temp.jpeg", preprocess=False))
    plt.imshow(cam, cmap='jet', alpha=0.3)
    plt.show()

def comp_loss(img,model):
    preds = model.predict(img)
    prob = round(np.amax(preds),2)
    return prob

def model(model_path):
    base_inception = InceptionV3(weights='imagenet', include_top=False, 
                                 input_shape=(299, 299, 3))
    model.summary()
    model_path_save = model_path
    #base_model = InceptionV3(weights='imagenet', include_top=False)
    #model = load_model(model_path_save)
    #return model
    
    #base_model = InceptionV3(weights='imagenet', include_top=False)
    #model = load_model(model_path_save)

    # load json and create model
    json_file = open(model_path_save, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    #model.load_weights(model_path_save)
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Loaded model from disk")
    return model





def image_resize_fil(path_img):
        im = Image.open(path_img)
        im = ImageEnhance.Brightness(im).enhance(0.7)
        #im = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
        im = ImageEnhance.Contrast(im).enhance(2.0)
        im.thumbnail((299, 299), Image.ANTIALIAS)
        im.save("img_temp.jpeg")
        im.show()


def predicccion(model_json, etiquetas, model_path_h5):

    # Cargar modelo
    model_oct = model(model_json)
    model_oct.load_weights(model_path_h5)
    #Cargar Imagen
    img = sys.argv[1]

    #imagen_resize y  Aplicar filtros
    image_resize_fil(img)

    #Cargar figura

    img = load_img("img_temp.jpeg")

    #calcular indice array mayor probabilidad
    ind = int(ind_pred(img,model_oct))

    # Predecir etiqueta
    predict_etiqueta = etiquetas[ind]

    # String predición
    etiqueta = "Predicción: " + predict_etiqueta


    # Predict prob si es menor de 0.90 rechazar
    threshold = 0.70
    proba = comp_loss(img,model_oct)
    if proba >= threshold:
        # Plot grad_cam OCT defecto
        plot_oct(grad_cam(model_oct,img,ind,'conv2d_88'), sys.argv[1], etiqueta)
       # print(proba)

    else:
        print("Defecto no detectado")
        #print(proba)
    
    print("defecto = ", predict_etiqueta, " Probabilidad = ", proba)

if __name__ == "__main__":

    H, W = 299, 299
    modelo1 = ['models/transfer_inceptionV3.json',['PNEMONIA', 'NORMAL'],'models/transfer_inceptionV3.h5']
    
    print(modelo1[0],modelo1[1],modelo1[2])
    predicccion(modelo1[0],modelo1[1],modelo1[2])

    # now = datetime.now()
    # current_time = now.strftime("%H:%M:%S")
    # print("Current Time =", current_time)    
    # l = 1
    # m = "modelo"
    # while l < 2:
    #    l = str(l)
    #    k = eval(m + l)
    #    predicccion(k[0],k[1],k[2])      
    #    l = int(l)
    #    l = 1 + l



