#!/bin/bash

# all classes
for f in *.tar; do
    tar xf $f
    d=`basename $f .tar`
    rm $f
done
