Data  preprocessing:

Dev: python preprocessing.py --data 'data/contest_Data'  --data_name content_Data  --input_file 'dev.csv' --data_type dev
Train: python preprocessing.py --data 'data/contest_Data'  --data_name content_Data  --input_file 'train.csv' --data_type train
Test: python preprocessing.py --data 'data/contest_Data'  --data_name content_Data  --input_file 'test.csv' --data_type Test



Generate dictionary with BERT encodings.

python bert-encoding.py --data data/contest_Data/Parsed_Data     --data_name contest

Run command :

RoBERT without  features:
python main.py --model_name RoBERT --data data/contest_Data/Parsed_Data --data_name contest --mem_dim 100  --input_dim  768 --num_classes 3 --run_type final --epochs 500 --max_num_sent 35 --file_len 5000 --feature_fname contest_train_merged_talo_feature.xlsx --domain_feature 0 --batchsize 10


RoBERT Without features:
python main.py --model_name doc_cls --data data/contest_Data/Parsed_Data --data_name contest --mem_dim 100  --input_dim  768 --num_classes 3 --run_type final --epochs 500 --max_num_sent 35 --file_len 5000 --feature_fname contest_train_merged_talo_feature.xlsx --domain_feature 0 --batchsize 10


RoBERT With features:
python main.py --model_name RoBERT --data data/contest_Data/Parsed_Data --data_name contest --mem_dim 100  --input_dim  768 --num_classes 3 --run_type final --epochs 500 --max_num_sent 35 --file_len 5000 --feature_fname train.xlsx --domain_feature 1 --batchsize 10

python main.py --model_name doc_cls --data data/contest_Data/Parsed_Data --data_name contest --mem_dim 100  --input_dim  768 --num_classes 3 --run_type final --epochs 500 --max_num_sent 35 --file_len 5000 --feature_fname train.xlsx --domain_feature 1 --batchsize 10


python load_model.py --model_name RoBERT --data data/contest_Data/Parsed_Data --data_name contest --mem_dim 100  --input_dim  768 --num_classes 3 --run_type final --epochs 500 --max_num_sent 35 --file_len 5000 --feature_fname test.xlsx --domain_feature 1 --batchsize 10 --expname train_447





python load_model.py --model_name doc_cls --data data/contest_Data/Parsed_Data --data_name contest --mem_dim 100  --input_dim  768 --num_classes 3 --run_type final --epochs 500 --max_num_sent 35 --file_len 5000 --feature_fname train.xlsx --domain_feature 1 --batchsize 10 --expname train_447


