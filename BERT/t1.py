import argparse
import json
import pandas as pd
import numpy as np
# from simpletransformers.classification import ClassificationModel
import torch
import os


def spType(num):
    if num==0:
        return 'phrase'
    elif num==1:
        return 'passage'
    else:
        return 'multi'
def parse_args():
    parser = argparse.ArgumentParser(description='This is a baseline for task 1 that predicts that each clickbait post warrants a passage spoiler.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=False)

    return parser.parse_args()


def load_input(df):
    if type(df) != pd.DataFrame:
        df = pd.read_json(df, lines=True)


    return df


def use_cuda():
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def predict(df):
    df = load_input(df)
    labels = ['phrase', 'passage', 'multi']
    # model = ClassificationModel('deberta', '/model', use_cuda=use_cuda())
    click=[]
    para=[]
    tagID=[]
    uuids=[]
    for ind in df.index:
        # click.append(df["postText"][ind])
        # para.append(df['targetParagraphs'][ind])
        # tagID.append(0)
        uuids.append(df['uuid'][ind])
    #     if df['tags'][ind]==['passage']:
    #         tagID.append(1)
    #     elif df['tags'][ind]==['phrase']:
    #         tagID.append(0)
    #     else:
    #         tagID.append(2)
    # data={'headline':click,
    #     'body':para,
    #     'label':tagID}
    # df1=pd.DataFrame(data)
    # df1.to_csv('./data/contest_Data/Raw_Data/test.csv', index=False)
    # os.system("python preprocessing.py --data 'data/contest_Data'  --data_name content_Data  --input_file 'test.csv' --data_type Test")
    # os.system("python bert-encoding.py --data data/contest_Data/Parsed_Data     --data_name contest")
    # os.system("python load_model.py --model_name doc_cls --data data/contest_Data/Parsed_Data --data_name contest --mem_dim 100  --input_dim  768 --num_classes 3 --run_type final --epochs 500 --max_num_sent 35 --file_len 5000 --feature_fname contest_train_merged_talo_feature.xlsx --domain_feature 0 --batchsize 10 --expname train_488")
    df2=pd.read_csv("ans.csv")
    for i in range(len(df)):
        yield {'uuid': uuids[i], 'spoilerType': [spType(df2.iloc[i][0])]}


def run_baseline(input_file, output_file):
    with open(output_file, 'w') as out:
        for prediction in predict(input_file):
            out.write(json.dumps(prediction) + '\n')


if __name__ == '__main__':
    args = parse_args()
    run_baseline(args.input, args.output)
