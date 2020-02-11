#!/bin/bash


path_pic="/home/jyosa/melanoma/Data/Images"

mal="malignant"
bien="benign"
if [[ -d "bening" && "malign" ]]
then
    rm -rf bening malign
    mkdir bening malign
else
    mkdir bening malign
fi

path_target=bening

for i in `ls /home/jyosa/melanoma/Data/Descriptions/ISIC*`
do
    tipo=`cat $i | grep "benign_malignant" | awk '{print $2}' | sed 's/"//g
                                                               s/,//g'`
    if [[ $tipo == $mal ]]
    then
       nombre=`echo $i | sed 's@/@ @g' | awk '{print $6}'`
       cp $path_pic/$nombre.jpeg malign
       echo "copiando " $nombre "al directorio bening, etiqueta =  " $tipo
    elif [[ $tipo == $bien ]]
    then
        nombre=`echo $i | sed 's@/@ @g' | awk '{print $6}'`
        cp $path_pic/$nombre.jpeg bening
        echo "copiando " $nombre "al directorio bening, etiqueta =  " $tipo 
    else
        echo "Sin clasificación el archivo no será guardado" 
    fi
done