#!/bin/sh
python train.py -f config/gru_config.json
python train.py -f config/bigru_atten2_config.json
python train.py -f config/rcnn_config.json
