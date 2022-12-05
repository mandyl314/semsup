#!/bin/bash

# data_dir="./class_descrs/newsgroups"
# data_dir="./class_descrs/class_descrs_spanish/newsgroups"
# output="./class_descrs/class_descrs_chinese/newsgroups"
for entry in "$data_dir"/*
do
    FILENAME=$(echo "$entry" | sed "s/.*\///")
    out_path="$output/$FILENAME"
    python3 translate.py "$entry" > $out_path

    # wc -l $entry
done