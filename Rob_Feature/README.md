# Text_Minor-at-CheckThat-2022
Official code for our submission for Task 3 at CheckThat! 2022
Data  preprocessing:

Dev: python preprocessing.py --data 'data/content_Data'  --data_name content_Data  --input_file 'dev.csv' --data_type Dev
Train: python preprocessing.py--data 'data/contest_Data'  --data_name content_Data  --input_file 'train.csv' --data_type Train
Test: python preprocessing.py --data 'data/contest_Data'  --data_name content_Data  --input_file 'test.csv' --data_type Test



Generate dictionary with BERT encodings.

python bert-encoding.py --data data/contest_Data/Parsed_Data     --data_name contest

Run command :

RoBERT with features:
python main.py --model_name RoBERT --data data/contest_Data/Parsed_Data --data_name contest --mem_dim 100  --input_dim  768 --num_classes 4 --run_type final --epochs 500 --max_num_sent 32 --file_len 5000 --feature_fname contest_train_merged_talo_feature.xlsx --domain_feature 1


RoBERT Without features:
python main.py --model_name RoBERT --data data/contest_Data/Parsed_Data --data_name contest --mem_dim 100  --input_dim  768 --num_classes 4 --run_type final --epochs 500 --max_num_sent 32 --file_len 5000 --feature_fname contest_train_merged_talo_feature.xlsx --domain_feature 0
