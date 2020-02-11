from tensorflow.keras import backend
import scipy as sp
import numpy as np
import pandas as pd
import skimage
import PIL
import scipy.ndimage as spi
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,Adagrad, Adadelta
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.utils.np_utils import to_categorical
import imageio
import model_evaluation_utils as meu
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
import argparse
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import model_from_json

# Esta función prepara un lote aleatorio del conjunto de datos.
def load_batch(dataset_df, batch_size = 25):
    batch_df = dataset_df.loc[np.random.permutation(np.arange(0,
                                                              len(dataset_df)))[:batch_size],:]
    return batch_df

# Esta función traza imágenes de muestra en un tamaño especificado y en una cuadrícula definida
def plot_batch(img_type, images_df, grid_width, grid_height, im_scale_x, im_scale_y):
    f, ax = plt.subplots(grid_width, grid_height)
    f.set_size_inches(12, 12)
    
    img_idx = 0
    for i in range(0, grid_width):
        for j in range(0, grid_height):
            ax[i][j].axis('off')
            ax[i][j].set_title(images_df.iloc[img_idx]['clase'][:10])
            ax[i][j].imshow(skimage.transform.resize(imageio.imread(DATASET_PATH + images_df.iloc[img_idx]['id'] + img_type),
                                             (im_scale_x,im_scale_y)))
            img_idx += 1
            
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.25)
    plt.show()



def datos_flo(tar_six,tar_siy,test_si,rand_sta,test_si2,rand_sta2):
    # cargar dataset
    train_data = np.array([img_to_array(load_img(img, target_size=(tar_six, tar_siy)))
                               for img in data_labels['image_path'].values.tolist()]).astype('float32')

    # crear  datasets de entrenamiento y test
    x_train, x_test, y_train, y_test = train_test_split(train_data, target_labels, 
                                                        test_size=test_si, 
                                                        stratify=np.array(target_labels), 
                                                        random_state=rand_sta)

    # crear datasets de entrenamiento y validacion  
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                      test_size=test_si2, 
                                                      stratify=np.array(y_train), 
                                                      random_state=rand_sta2)
    return train_data, x_train, x_test, y_train, y_test, x_val, y_val



def data_gen(BATCH_SIZE, rot_ran, width_s_r, height_s_r, hor_flip, seed):
    #Generacion de imágenes a partir de las originales
    BATCH_SIZE = 32

    # generador.
    train_datagen = ImageDataGenerator(rescale=1./255, 
                                       rotation_range=rot_ran, 
                                       width_shift_range=width_s_r,
                                       height_shift_range=height_s_r, 
                                       horizontal_flip = hor_flip)
    train_generator = train_datagen.flow(x_train, y_train_ohe, shuffle=False, 
                                     batch_size=BATCH_SIZE, seed=seed)
                                     
    # Crear generador de validación
    val_datagen = ImageDataGenerator(rescale = 1./255)
    val_generator = train_datagen.flow(x_val, y_val_ohe, shuffle=False, 
                                       batch_size=BATCH_SIZE, seed=1)
    return train_datagen, train_generator, val_datagen, val_generator


def tranf_learn(pesos,shapex,shapey,shapez,activat,activat2,loss,learning_rate,moment,BATCH_SIZE,epochs,save_file_path,save_json):
    # Obtener el modelo InceptionV3 para que podamos realizar el aprendizaje de transferencia
    base_inception = InceptionV3(weights=pesos, include_top=False, 
                                 input_shape=(shapex, shapey, shapez))
    

    # Agregamos 
    out = base_inception.output
    out = GlobalAveragePooling2D()(out)
    #out = Dense(512, activation='relu')(out)
    #out = Dense(512, activation='relu')(out)
    out = Flatten()(out)
    out = Dense(1024, activation="relu")(out)
    out = BatchNormalization()(out)
    out = Dropout(0.5)(out)
    out = Dense(512, activation="relu")(out)
    out = BatchNormalization()(out)
    out = Dropout(0.5)(out)
    out = Dense(512, activation="relu")(out)
    out = BatchNormalization()(out)
    out = Dropout(0.5)(out)
    out = Dense(512, activation="relu")(out)
    

    
    total_classes = y_train_ohe.shape[1]
    


    #Este es un problema de clasificación binaria, por lo que utilizamos 
    # la función de activación sigmoidea en la capa de salida.

    predictions = Dense(total_classes, activation=activat2)(out)

    model = Model(inputs=base_inception.input, outputs=predictions)

    opt1 =  optimizers.SGD(lr=learning_rate, momentum=moment, nesterov=True)
    opt2 =  Adadelta(lr=learning_rate, rho=0.95)
    opt3 =  Adagrad(lr=0.0001)


    # Compile 
    model.compile(loss=loss, optimizer=opt1, metrics=["accuracy"])

    model.summary()



    # Entrenar modelo
    batch_size = BATCH_SIZE
    train_steps_per_epoch = x_train.shape[0] // batch_size
    val_steps_per_epoch = x_val.shape[0] // batch_size

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_steps_per_epoch,
                                  validation_data=val_generator,
                                  validation_steps=val_steps_per_epoch,
                                  epochs=epochs, verbose=1)


    # serialize model to JSON
    model_json = model.to_json()
    with open(save_json, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(save_file_path)
    print("Saved model to disk")

    #model.save(save_file_path)
    return history

def plot_eval(total_epchs,plot_name,space):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    t = f.suptitle('Inception V3 Evaluación', fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)
    epoch_list = list(range(1,total_epchs))
    ax1.plot(epoch_list, history.history['acc'], label='Precisión Entrenamineto')
    ax1.plot(epoch_list, history.history['val_acc'], label='Precisión Validación')
    ax1.set_xticks(np.arange(0, total_epchs, space))
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")
    ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
    ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
    ax2.set_xticks(np.arange(0, total_epchs, space))
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")
    plt.savefig(plot_name)
    plt.show()


if __name__ == "__main__":


    # Argumentos
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epochs", type=int, required=True,
        help="Número de epochs para entrenamiento")
    ap.add_argument("-p", "--path", type=str, required=True,
        help="Path directorio imagenes ej. '/home/jyosa/all_images'")
    ap.add_argument("-l", "--labels", type=str, required=True,
        help="Path archivo labels.csv ej. '/home/jyosa/labels.csv'")
    ap.add_argument("-ex", "--ext", type=str, required=True,
        help="Tipo de imágen. Ejemplo '.jpeg'")
    args = vars(ap.parse_args())


    np.random.seed(42)

    #si no tiene los datos etiquetados en un archivo diferente use get_labels.py

    DATASET_PATH = args["path"]
    LABEL_PATH = args["labels"]

    # cargar el conjunto de datos y visualizar datos de muestra
    dataset_df = pd.read_csv(LABEL_PATH)
    batch_df = load_batch(dataset_df, batch_size=36)
    plot_batch(args["ext"], batch_df, grid_width=6, grid_height=6,
        im_scale_x=64, im_scale_y=64)


    #mirando cómo se ven las etiquetas del conjunto de datos para tener una idea de todas la posible eqtiquetas.

    data_labels = pd.read_csv(LABEL_PATH)
    target_labels = data_labels['clase']
    print("Etiquetas encontradas: ", len(set(target_labels)))
    data_labels.head()

    #Lo que hacemos a continuación es agregar la ruta de imagen exacta para cada
    # imagen presente en el disco usando el siguiente código. Esto nos ayudará a 
    # localizar y cargar fácilmente las imágenes durante el entrenamiento del modelo.


    train_folder = DATASET_PATH
    data_labels['image_path'] = data_labels.apply(lambda row: (train_folder + row["id"] + args["ext"] ), 
                                            axis=1)
    data_labels.head()


    #Preparar  conjuntos de datos de entrenamiento, prueba y validación.

    #Parámetros
    target_size_x = 299
    target_size_y = 299
    test_size = 0.3
    random_state = 42
    test_size2 = 0.15
    random_state2 = 42


    train_data, x_train, x_test, y_train, y_test, x_val, y_val = datos_flo(target_size_x,target_size_y,test_size,random_state,test_size2,random_state2)

    print('Tamaño inicial del conjunto de datos:', train_data.shape)
    print('Tamaño inicial de conjuntos de datos de prueba y entrenamiento:', x_train.shape, x_test.shape)
    print('Tamaño de conjuntos de datos de entrenamiento y validación:', x_train.shape, x_val.shape)
    print('Tamaño de conjuntos de datos de entrenamiento, prueba y validación:\n', x_train.shape, x_test.shape, x_val.shape)

    #conviertir las etiquetas de clase de texto en etiquetas codificadas one-hot


    y_train_ohe = pd.get_dummies(y_train.reset_index(drop=True)).values
    y_val_ohe = pd.get_dummies(y_val.reset_index(drop=True)).values
    y_test_ohe = pd.get_dummies(y_test.reset_index(drop=True)).values
    print(y_train_ohe.shape, y_test_ohe.shape, y_val_ohe.shape)

    #Parámetros
    batch_size = 32
    rotation_range = 30
    width_shift_range = 0.2
    height_shift_range = 0.2
    horizontal_flip = 'True'
    seed = 25

    train_datagen, train_generator, val_datagen, val_generator = data_gen(batch_size, rotation_range, width_shift_range, height_shift_range, horizontal_flip, seed)

    #Transfer Learning with Google’s Inception V3 Model

    #Parámetros

    weights = 'imagenet'
    input_shapex = 299
    input_shapey = 299
    input_shapez = 3
    activation = 'relu'
    activation_pred = 'sigmoid'
    loss = "binary_crossentropy"
    learning_rate = 0.00005 
    momentum = 0.95
    batch_size = 32
    epochs = args["epochs"]
    model_path_save = 'models/transfer_inceptionV3.h5'
    model_path_save_json = 'models/transfer_inceptionV3.json'

    history = tranf_learn(weights,input_shapex,input_shapey,input_shapez,activation,activation_pred,loss,learning_rate,momentum,batch_size,epochs,model_path_save,model_path_save_json)


    # Evaluación Inception V3

    #Parámetros
    num_epochs = epochs + 1
    Plot_name = 'Permormance_1.png'
    space = 50

    plot_eval(num_epochs,Plot_name,space)



    #Evaluación del modelo


    base_model = InceptionV3(weights='imagenet', include_top=False)

    # cargar json y crear modelo
    json_file = open(model_path_save_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # cargar los pesos dentro del nuevo modelo
    model.load_weights(model_path_save)
    print("cargando modelo al disco")



    # escalando características
    x_test /= 255.

    # predicciones
    test_predictions = model.predict(x_test)


    labels_ohe_names = pd.get_dummies(target_labels, sparse=True)
    predictions = pd.DataFrame(test_predictions, columns=labels_ohe_names.columns)
    predictions = list(predictions.idxmax(axis=1))
    test_labels = list(y_test)

    #evaluación del modelo
    meu.get_metrics(true_labels=test_labels, 
                    predicted_labels=predictions)

    meu.display_classification_report(true_labels=test_labels, 
                                    predicted_labels=predictions, 
                                    classes=list(labels_ohe_names.columns))
    print(meu.display_confusion_matrix_pretty(true_labels=test_labels, 
                                    predicted_labels=predictions, 
                                    classes=list(labels_ohe_names.columns)))
  



    grid_width = 5
    grid_height = 5
    f, ax = plt.subplots(grid_width, grid_height)
    f.set_size_inches(15, 15)
    batch_size = 25
    dataset = x_test

    labels_ohe_names = pd.get_dummies(target_labels, sparse=True)
    labels_ohe = np.asarray(labels_ohe_names)
    label_dict = dict(enumerate(labels_ohe_names.columns.values))
    model_input_shape = (1,)+model.get_input_shape_at(0)[1:]
    random_batch_indx = np.random.permutation(np.arange(0,len(dataset)))[:batch_size]

    img_idx = 0
    for i in range(0, grid_width):
        for j in range(0, grid_height):
            actual_label = np.array(y_test)[random_batch_indx[img_idx]]
            prediction = model.predict(dataset[random_batch_indx[img_idx]].reshape(model_input_shape))[0]
            label_idx = np.argmax(prediction)
            predicted_label = label_dict.get(label_idx)
            conf = round(prediction[label_idx], 2)
            ax[i][j].axis('off')
            ax[i][j].set_title('Actual: '+actual_label+'\nPred: '+predicted_label + '\nConf: ' +str(conf))
            ax[i][j].imshow(dataset[random_batch_indx[img_idx]])
            img_idx += 1

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.5, hspace=0.55) 
    plt.show()