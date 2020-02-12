# Aprendizaje por transferencia para clasificación de imágenes diagnósticas

Como este es un tutorial, no pretende dar conceptos avanzados sobre el aprendizaje por transferencia, ni tampoco sobre redes neuronales profundas, o redes neuronales convolucionales, pero puede tomar los cursos gratuitos de EDX, o consultar las siguiente páginas: http://cs231n.github.io/transfer-learning/ , https://medium.com/analytics-vidhya/transfer-learning-with-convolutional-neural-networks-e9c513d41506 , https://www.sciencedirect.com/science/article/abs/pii/S0031320319303516 .

## Datos

Cualquier proyecto de aprendizaje de máquina debe iniciar por la obtención de los datos, de hecho google hace algún tiempo lanzó su buscador exclusivo para colección de datos (dele una ojeada: https://datasetsearch.research.google.com/). Para este caso particular, vamos a requerir imágenes diagnósticas. El diagnóstico por imágenes permite a los médicos observar el interior del cuerpo para buscar indicios sobre una condición médica específica, hay de muchos tipos, pero específicamente vamos a concentrarnos en la neumonía. La neumonía es una infección que causa inflamación en uno o ambos pulmones y puede ser causada por un virus, bacterias, hongos u otros gérmenes.
Un médico puede realizar un examen físico y usar una radiografía de tórax, una tomografía computarizada de tórax, una ecografía de tórax o una biopsia pulmonar con aguja para ayudar a diagnosticar su afección. El médico puede evaluar más su condición y función pulmonar mediante toracocentesis, colocación de sonda torácica o drenaje de absceso guiado por imagen.

### Estadísticas de neumonía en niños Colombianos

La Fundación Neumológica Colombiana reveló que 920.136 menores de edad fallecieron en 2015 por este padecimiento respiratorio, lo que supone el 15 % de las muertes de niños menores de cinco años en el mundo. (fuente: http://www.redmas.com.co/salud/neumonia-muerte-colombia-enfermedad-842904/), Neumologos de la Fundación Neumológica de Colombia aseguran que esta enfermedad está entre las tres primeras causas de muerte en la niñez del país. “Aunque la situación es grave y las cifras alarmantes, la neumonía se puede evitar con prevención diaria y en caso de ser necesario con tratamiento médico“

La neumonía puede ser difícil de diagnosticar debido a que puede parecer un resfriado o un cuadro gripal. De hecho muchas personas no se dan cuenta que es más grave, hasta que dura más que un resfriado o gripe común.
Una de las pruebas para diagnosticar la neumonía es por medio de una radiografía de tórax. La radiografía de tórax es una prueba indolora que obtiene imágenes de las estructuras internas del tórax, como el corazón, los pulmones y los vasos sanguíneos. Una radiografía del pecho es la mejor prueba para el diagnóstico de neumonía junto a la clínica. Sin embargo, esta prueba no le dirá a su médico qué tipo de germen está causando la neumonía. 

Asi pues, el proceso puede durar varias horas (días en muchos casos) hasta que un especialista neumólogo revise los rayos X, pero que tal si un sistema puede diagnósticar si una persona, especialmente niños puede tener neumonía bacteriana o viral, y darle prioridad para que el neumólogo realice una consulta prioritaria?
Pues bien es lo que vamos a intentar hacer aquí.

Los datos los vamos a obtener en el siguiente link: https://data.mendeley.com/datasets/rscbjbr9sj/2/files/f12eaf6d-6023-432f-acc9-80c9d7393433.

El archivo trae  dos directorios: test y train, en el directorio train se encuentran alrededor de 5000 imágenes de tórax resueltas por rayos X, de personas normales y personas con neumonía.


![alt text](https://github.com/jyosa/transfer-learning-for-diagnostic-images-spanish-/blob/master/NORMAL-1.jpeg)

Persona normal

![alt text](https://github.com/jyosa/transfer-learning-for-diagnostic-images-spanish-/blob/master/PNEUMONIA-1.jpeg)

Persona con neumonía

Como puede observar de las imágenes anteriores, nuestra red neuronal tiene que aprender a diferenciar entre una imagen de rayos X normal y una con neumonía.

Hay muchas formas de trabajar con imágenes y hay cientos de tutoriales en la web con múltiples formas de procesar los píxeles de una imagen, asi que sientase libre de explorar otras formas, a lo mejor obtienen mejores resultados.

## Cuántas imágenes requiere una red neuronal?

5000 imágenes pueden ser muy pocas para una red neuronal convolucional, de hecho, proyectos de investigación llevados a cabo por NVIDIA han entrenado alrededor de unas 100 mil imágenes diagnósticas de OCT (optical coherence tomography), con excelentes resultados (por cierto con transfer learning hice lo mismo con los mismos resultados y unas 1500 fotos aproximadamente). Como ya se imaginan el éxito del transfer learning esta en que no se requieren muchas imágenes para entrenar una red neuronal. Ahora bien esto tiene unos pros y contras. Algo que no nos favorece en este caso (contra), es que los modelos usados para el aprendizaje por transferencia como; VGG16; VGG19; ResNet50; Inception v3, han sido entrenados con muchas imágenes, de muchos tipos, pero no con el tipo de imagen que queremos clasificar aquí, así que como se han dado cuenta estamos en el escenario donde el conjunto de datos de destino es grande y diferente del conjunto de datos de entrenamiento base.
Asi que como el conjunto de datos de destino es grande y diferente del conjunto de datos base, podemos entrenar el modelo base desde cero, vamos a usar una técnica llamada fine tunning (ajuste fino),El ajuste fino requiere que no solo actualicemos la arquitectura CNN sino que también la entrenemos para aprender nuevas clases de objetos, como es nuestro caso. (https://www.pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/). Sin embargo, en la práctica, es beneficioso inicializar los pesos de la red pre-entrenada y ajustarlos, ya que podría acelerar el entrenamiento, pero vamos a hacerlo desde cero, ya que aunque el conjunto de datos de destino es grande, no es tan considerable, lo cual reduce el tiempo de cómputo a un par de horas con una GPU NVIDIA 1080  (creanme ya lo hice ;))

## Preparando imágenes

Antes que nada asegúrese que tiene instaladas las librerías en las versiones correctas:

* python                    3.6.9
* keras-gpu                 2.2.4
* numpy                     1.17.3
* opencv                    3.4.2
* scipy                     1.3.1
* tensorflow                1.14.0

Para hacer la evaluación del modelo es necesario la siguiente librería;  [model_evaluation_utils.py](https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/notebooks/Ch05_Building_Tuning_and_Deploying_Models/model_evaluation_utils.py), coloquelo en el directorio maestro.

Una de las cosas que más me da escozor en un tutorial, es cuando se toman directamente datos de paquetes como keras, por ejemplo cifar10. El problema de empezar un tutorial asi es que  alguien que está aprendiendo necesita saber cómo diablos proceso mis datos para ser integrados a la red neuronal!!!
Así que eso es lo que vamos a hacer aquí. Vamos a crear dos directorios para colocar en ellos, una muestra de imágenes normal y con neumonía:

```
$ mkdir pneumonia normal
```
Ahora use el script llamado trans.sh para transladar 1500 imágenes normales y 1500 con neumonía, cambie el path en el script con su propio path. La idea es tener la misma cantidad de imágenes de ambas etiquetas (normales y con neumonía). 

El siguiente paso es poner filtros a las imágenes, etiquetarlas y por último poner todas las imágenes procesadas (normales y con neumonía), e un mismo directorio, así que va a correr el script (prepare_pict.py), dos veces uno para las imágenes normales y el siguiente para las imágenes con neumonía. Primero vamos a crear un tercer directorio donde vamos a poner todas las imágenes procesadas:

```
$ mkdir clasificacion
```
Luego vamos a cambiar el script, colocando el path del nuevo directorio (clasificacion) y vamos a cambiar el path de donde vamos a tomar las imágenes, que son pneumonia y normal (recuerde por cada uno hay que correr el script)

```
from PIL import Image, ImageFilter,ImageEnhance
import os


directory = '/home/jyosa/melanoma/chest_xray/train/pneumonia/'  ← Modificar
dir_target = '/home/jyosa/melanoma/clasificacion/' ← Modificar
name = "PNEUMONIA" ← modificar

num = 0
for filename in os.listdir(directory):
    if filename.endswith(".jpeg"):
        filename1 = str(directory + filename)
        im = Image.open(filename1)
        im = ImageEnhance.Brightness(im).enhance(0.7)
        im = ImageEnhance.Contrast(im).enhance(2.0)
        im.thumbnail((299, 299), Image.ANTIALIAS)
        num = int(num)
        num = num +1
        if num == 10000:
            break
        num = str(num)
        file_tar = dir_target + name + "-" + num + ".jpeg"
        
        im.save(file_tar)
```

Y para las normales:

```
from PIL import Image, ImageFilter,ImageEnhance
import os

directory = '/home/jyosa/melanoma/chest_xray/train/normal/'  ← Modificar
dir_target = '/home/jyosa/melanoma/clasificacion/' ← Modificar
name = "NORMAL" ← modificar

num = 0
for filename in os.listdir(directory):
    if filename.endswith(".jpeg"):
        filename1 = str(directory + filename)
        im = Image.open(filename1)
        im = ImageEnhance.Brightness(im).enhance(0.7)
        im = ImageEnhance.Contrast(im).enhance(2.0)
        im.thumbnail((299, 299), Image.ANTIALIAS)
        num = int(num)
        num = num +1
        if num == 10000:
            break
        num = str(num)
        file_tar = dir_target + name + "-" + num + ".jpeg"
        
        im.save(file_tar)
```

```
$ python prepare_pict.py
```
Si hace un recorrido por las imágenes transformadas, se va a dar cuenta que hemos aplicado filtros de contraste y brillo.


![alt text](https://github.com/jyosa/transfer-learning-for-diagnostic-images-spanish-/blob/master/person1_bacteria_1.jpeg)

Sin filtros

![alt text](https://github.com/jyosa/transfer-learning-for-diagnostic-images-spanish-/blob/master/PNEUMONIA-1.jpeg)

Con filtros


## Generar etiquetas

Es necesario generar las etiquetas (NORMAL, PNEUMONIA) para entrenar nuestra red neuronal, para tal fin vamos a crear un archivo tipo csv, para que sea más fácil integrar las etiquetas con las imágenes. Para tal fin hay que correr el script llamado get_labels.py, hay que editarlo con el path donde ha colocado el directorio clasificacion y el path donde quiere guardar el archivo labels.csv


```
import glob
import numpy as np
import os
import shutil
import glob
import numpy as np


 
filePath = '/home/jyosa/melanoma/labels.csv' ← editar
 
# As file at filePath is deleted now, so we should check if file exists or not not before deleting them
if os.path.exists(filePath):
    os.remove(filePath)
else:
    print("Can not delete the file as it doesn't exists")


path = 'clasificacion/' ← editar
save_path =  os.getcwd() + "/" + 'labels.csv'
with open(save_path,'a+') as data_file:

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpeg' in file:
                files.append(os.path.join(r, file))
    data_file.write("id,clase")    
    for f in files:
        
        id_img=f.split('/')[1].split('.')[0].strip()
        clase=f.split('/')[1].split('-')[0].strip()
        print(clase)
        data_file.write("\n{},{}".format(id_img,clase))
    
```

```
$ python get_labels.py
```

## Entrenamiento y evaluación de la red neuronal

Ya tenemos todo listo para entrenar nuestra red neuronal, para tal fin vamos a usar como modelo base Inception V3 de google, aquí más información (https://medium.com/@sh.tsang/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c).

El script de python llamado binary.py, realiza una clasificación binaria, esto es va a clasificar entre pacientes con neumonía y normales. Antes de correr el script siéntase libre de recorrerlo y observar cada parte qué hace específicamente, los pasos son los siguientes:

cargar el conjunto de datos y visualizar datos de muestra.
Observar cómo se ven las etiquetas del conjunto de datos para tener una idea de todas la posible etiquetas.
Lo que hacemos a continuación es agregar la ruta de imagen exacta para cada imagen presente en el disco. Esto nos ayudará a localizar y cargar fácilmente las imágenes durante el entrenamiento del modelo.
Preparar conjuntos de datos de entrenamiento, prueba y validación.
convertir las etiquetas de clase de texto en etiquetas codificadas one-hot
Data augmentation: La idea aquí es que, cuando se dispone de un número de imágenes relativamente pequeño, podemos aumentar el número modificando las imágenes originales (haciendo zoom, escalado, flip horizontal, etc).
Transferir aprendizaje con el modelo Inception V3 de Google.

Una vez finaliza el ciclo de entrenamiento se guardan dos archivos en el directorio model (json y h5), esto es importante si quiere que el modelo final corra en algún tipo de aplicación y el deploy sea má rápido. 
Y último paso se hace una validación del modelo.


Finalmente es hora de correr todo el script binary.py:

```
$ python binary_class.py -e 60 -p clasificacion/ -l labels.scv -ex .jpeg
```
```
-e = número de epochs
-p = directorio donde se encuentran las imágenes procesadas (nota: no olvide colocar / al final).
-l = archivo donde hemos puesto nuestras etiquetas
-ex = tipo de archivo, para este caso .jpeg, pero puede funcionar con otro tipo de extensión, jpg, png etc.
```

## Resultados

Aquí algunos resultados:
```

              precision    recall  f1-score   support

      NORMAL       0.93      0.99      0.96       405
   PNEUMONIA       0.99      0.93      0.96       450

    accuracy                            0.96       855
   macro avg        0.96      0.96      0.96       855
weighted avg       0.96       0.96      0.96       855



Matriz de confusión

                  Predicted:          
                           NORMAL        PNEUMONIA
Actual: NORMAL              401                 4
        PNEUMONIA            32                418
```

![alt text](https://github.com/jyosa/transfer-learning-for-diagnostic-images-spanish-/blob/master/Permormance_1.png)

![alt text](https://github.com/jyosa/transfer-learning-for-diagnostic-images-spanish-/blob/master/results_2.png)


Sin una optimización de hiperparametros se obtiene un sorprendente 96% de precisión.





