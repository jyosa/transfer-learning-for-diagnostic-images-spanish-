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
from tensorflow.keras.models import model_from_json



def load_img(imag_path):
    img_path = imag_path
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x /= 255.
    x = np.expand_dims(x, axis=0)
    return x

def grad_cam(x,img_path,lab_pred):
    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer('conv2d_93')
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(192):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    probabilidad = round(np.amax(preds)*100,2)
    #label_fig = 'Predicción : ' + str(lab_pred) + "  " +  str(probabilidad) + str('%')
    label_fig = 'Predicción : ' + str(lab_pred) 
    plt.figure(figsize=(24,12))
    cv2.imwrite('./tmp.jpg', superimposed_img)
    plt.imshow(mpimg.imread('./tmp.jpg'))
    plt.title(label_fig)
    plt.show()


json_file = open('models/transfer_inceptionV3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
#model.load_weights(model_path_save)
model.load_weights('models/transfer_inceptionV3.h5')
print("Loaded model from disk")


model_path_save = 'models/cnn_transfer_inceptionV3.h5'
base_model = InceptionV3(weights='imagenet', include_top=False)
#model = load_model(model_path_save)
model.summary()
etiquetas = ['DRUSEN', 'NORMAL']

img = sys.argv[1]
arr_image = load_img(img)
preds = model.predict(arr_image)
ind = preds.argmax()
predict_etiqueta = etiquetas[ind]
predict_etiqueta = etiquetas[ind]
print(predict_etiqueta)
grad_cam(arr_image, img, predict_etiqueta)

