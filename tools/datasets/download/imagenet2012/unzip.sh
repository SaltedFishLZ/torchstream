#!/bin/bash

# training set

mkdir train
tar -xvf ILSVRC2012_img_train.tar -C train/

for f in *.tar; do
    d=`basename $f .tar`
    mkdir $d
    (cd $d && tar xf ../$f)
    echo $d
    rm -r $f
done
rm -r *.tar
