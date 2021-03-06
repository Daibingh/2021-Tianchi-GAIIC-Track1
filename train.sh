#!/bin/sh
mkdir -p logging/task1
mkdir -p logging/task2
mkdir -p logging/pretrain
rm -rf saving
mkdir -p saving/task1
mkdir -p saving/task2
mkdir -p saving/pretrain

echo "start bert pretraining ..."
python pretrain_bert.py -f config/pretrain/bert_pretrain.json 


echo "start training ..."

python train.py -f config/task1/bert.json --random_seed 937 --n_fold 10 --pred_W 10 
python train.py -f config/task1/gru.json --random_seed 937 --n_fold 40
python train.py -f config/task1/bigru_atten2.json --random_seed 904 --n_fold 40
python train.py -f config/task1/seq2seqatten2.json --random_seed 1039 --n_fold 40
python train.py -f config/task1/rcnn.json --random_seed 503 --n_fold 40

python train.py -f config/task2/bert.json --random_seed 937 --n_fold 10 --pred_W 10 
python train.py -f config/task2/gru.json --random_seed 937 --n_fold 40 
python train.py -f config/task2/bigru_atten2.json --random_seed 904 --n_fold 40
python train.py -f config/task2/seq2seqatten2.json --random_seed 1039 --n_fold 40
python train.py -f config/task2/rcnn.json --random_seed 503 --n_fold 40 
