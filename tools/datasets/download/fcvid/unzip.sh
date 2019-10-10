#!/bin/bash

# all classed

for f in *.tar; do
    d=`basename $f .tar`
    mkdir $d
    (cd $d && tar xf ../$f)
    echo $d
    rm -r $f
done
rm -r *.tar
