#!/bin/sh
mkdir -p logging/task1
mkdir -p logging/task2
rm -rf saving/task1/*
rm -rf saving/task2/*

echo "start training ..."
python train.py -f config/task1/gru.json --random_seed 728
python train.py -f config/task1/bigru_atten2.json --random_seed 728
python train.py -f config/task1/rcnn.json --random_seed 728

python train.py -f config/task2/gru.json --random_seed 728
python train.py -f config/task2/bigru_atten2.json --random_seed 728
python train.py -f config/task2/rcnn.json --random_seed 728
