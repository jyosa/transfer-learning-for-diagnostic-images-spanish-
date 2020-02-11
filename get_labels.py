import glob
import numpy as np
import os
import shutil
import glob
import numpy as np


 
filePath = '/home/jyosa/melanoma/labels.csv'
 
# As file at filePath is deleted now, so we should check if file exists or not not before deleting them
if os.path.exists(filePath):
    os.remove(filePath)
else:
    print("Can not delete the file as it doesn't exists")


path = 'pneumonia/'
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
    
