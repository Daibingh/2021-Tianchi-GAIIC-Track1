#!/bin/sh
rm result*.csv
mkdir data
python preprocess.py
python utils/word2vec_pretain.py 

sh train.sh

python utils/gen_model_list.py saving/task1
python utils/gen_model_list.py saving/task2

echo "starting predict ..."
python test.py \
		--out_file "result_part1.csv" \
		--model_list_file "saving/task1/model_list.csv" \
		--test_file "data/rd2_testA.csv"
python test.py \
		--out_file "result_part2.csv" \
		--model_list_file "saving/task2/model_list.csv" \
		--test_file "data/rd2_testA.csv"
python merge.py
