Use the codes in pre.py file to generate .csv file as shown in Raw_Data folder under Data folder.

## Data  preprocessing:

Dev: ```python preprocessing.py --data 'data/contest_Data'  --data_name content_Data  --input_file 'dev.csv' --data_type dev```
Train: ```python preprocessing.py --data 'data/contest_Data'  --data_name content_Data  --input_file 'train.csv' --data_type train```
Test: ```python preprocessing.py --data 'data/contest_Data'  --data_name content_Data  --input_file 'test.csv' --data_type Test```



## Generate dictionary with BERT encodings.

```python bert-encoding.py --data data/contest_Data/Parsed_Data     --data_name contest```


## Model Training

RoBERT Without features:
Similarity:
```python main.py --model_name RoBERT --data data/contest_Data/Parsed_Data --data_name contest --mem_dim 100  --input_dim  768 --num_classes 3 --run_type final --epochs 500 --max_num_sent 35 --file_len 5000 --feature_fname contest_train_merged_talo_feature.xlsx --domain_feature 0 --batchsize 10```

Document Classification:
```ython main.py --model_name doc_cls --data data/contest_Data/Parsed_Data --data_name contest --mem_dim 100  --input_dim  768 --num_classes 3 --run_type final --epochs 500 --max_num_sent 35 --file_len 5000 --feature_fname contest_train_merged_talo_feature.xlsx --domain_feature 0 --batchsize 10```

For loading the model:
```python load_model.py --model_name RoBERT --data data/contest_Data/Parsed_Data --data_name contest --mem_dim 100  --input_dim  768 --num_classes 4 --run_type final --epochs 500 --max_num_sent 35 --file_len 5000 --feature_fname contest_train_merged_talo_feature.xlsx --domain_feature 0 --batchsize 10 --expname train_488```


Now, a file named "ans.csv" would have been generated with the labels.

Run the following command for **Task 1**:
```python t1.py --input input.jsonl --output task1.jsonl```

Run the following command for **Task 2**:
```python t2.py --input input.jsonl  --output task2.jsonl```



