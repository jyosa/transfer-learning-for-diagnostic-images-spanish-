#!/bin/bash

ini=1
fin=1501
path1=/home/jyosa/melanoma/chest_xray/train/PNEUMONIA
path2=/home/jyosa/melanoma/chest_xray/train/PNEUMONIA1500
cd $path1

ls -1 *.jpeg > /home/jyosa/melanoma/lista

while [ $ini != $fin ]
do 
    file=`cat /home/jyosa/melanoma/lista | awk '{if (NR=='$ini') print $1}'`
    echo "$path1 .... $ini"
    echo "traferring $file...... to $file $path2/$file"
    cp $path1/$file $path2/$file
    pwd

    let ini=$ini+1
done
