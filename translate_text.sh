#!/bin/bash

# data_dir=./class_descrs/cifar/
data_dir=./class_descrs/class_descrs_spanish/cifar
# output="./class_descrs/class_descrs_spanish/awa"
for entry in "$data_dir"/*
do
    # FILENAME=$(echo "$entry" | sed "s/.*\///")
    # out_path="$output/$FILENAME"
    # python3 translate.py "$entry" > $out_path
    wc -l $entry
done