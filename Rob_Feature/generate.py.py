import argparse
import json
import pandas as pd
import numpy as np
# from simpletransformers.classification import ClassificationModel
import torch
import os
import nltk

# Install the punkt tokenizer
nltk.download('punkt')
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from nltk.tokenize import sent_tokenize

from scipy.spatial.distance import cosine
from numpy import dot
from numpy.linalg import norm


from sentence_transformers import SentenceTransformer
models= SentenceTransformer('all-MiniLM-L6-v2')
# !pip install torch==1.4.0
from transformers import DebertaTokenizer, DebertaModel
import torch
tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
modeld = DebertaModel.from_pretrained("microsoft/deberta-base")
import torch
from num2words import num2words
from rank_bm25 import BM25Okapi
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
def remove_stopwords(words):
    return [word for word in words if word not in stop_words]

def stemmerw(words):
    return [stemmer.stem(word) for word in words]

def lemma(words):
    return [lemmatizer.lemmatize(word) for word in words]
def tken(corpus):
    temp_corpus= [doc.split(" ") for doc in corpus]
    tokenized_corpus=[]
    for sen in temp_corpus:
        sen=remove_stopwords(sen)
#         sen=stemmerw(sen)
#         sen=lemma(sen)
        tokenized_corpus.append(sen)
    return tokenized_corpus

import re
def remove_pun(text):
    res= re.sub(r'[-"()\#@.,‘:?!]', ' ', text)
    # remove additional space from string
    res = re.sub(' +', ' ', res)
    #res = res.lower()
    return res

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
    pra=[]
    # tagID=[]
    uuids=[]
    for ind in df.index:
        click.append(df["postText"][ind])
        pra.append(df['targetParagraphs'][ind])
        # tagID.append(0)
        uuids.append(df['uuid'][ind])
    #     if df['tags'][ind]==['passage']:
    #         tagID.append(1)
    #     elif df['tags'][ind]==['phrase']:
    #         tagID.append(0)
    #     else:
    #         tagID.append(2)
    # data={'headline':click,
    #     'body':pra,
    #     'label':tagID}
    # df1=pd.DataFrame(data)
    # df1.to_csv('./data/contest_Data/Raw_Data/test.csv', index=False)
    # os.system("python preprocessing.py --data 'data/contest_Data'  --data_name content_Data  --input_file 'test.csv' --data_type Test")
    # os.system("python bert-encoding.py --data data/contest_Data/Parsed_Data     --data_name contest")
    # os.system("python load_model.py --model_name doc_cls --data data/contest_Data/Parsed_Data --data_name contest --mem_dim 100  --input_dim  768 --num_classes 4 --run_type final --epochs 500 --max_num_sent 35 --file_len 5000 --feature_fname contest_train_merged_talo_feature.xlsx --domain_feature 0 --batchsize 10 --expname train_488")
    df2=pd.read_csv("ans.csv")
    print(df2)
    answer=[]
    model_name = "deepset/roberta-base-squad2"
    for index in range(len(df)):
        print(df2.iloc[index][0])
        if df2.iloc[index][0]==0:
            clickbait=click[index][0]
            para=pra[index]
            sent=''
            for p in para:
                sent+=p
                #roberta
            nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
            QA_input = {
                'question': clickbait,
                'context': sent
            }
            res = nlp(QA_input)
            answer.append([res['answer']])
        else:
            rankings=[]
            dic={}
            para=pra[index]
            clickbait=click[index]
            sent=[]
            for sentences in para:
                temp=sent_tokenize(sentences)
                for tempo in temp:
                    sent.append(tempo)
            embeddings=models.encode(sent)
            embed_clickbait=models.encode(clickbait)
            for sentence, embedding in zip(sent, embeddings):
                # Compute the cosine similarity between the first and second sentence
                a=embed_clickbait
                b=embedding
                similarity=dot(a, b)/(norm(a)*norm(b))
                dic[sentence]=similarity
            tem=sent
            sbert=sorted(tem,key=lambda x:dic[x], reverse=True)
            rankings.append(sbert)
            #deberta
            inputs = tokenizer(clickbait, return_tensors="pt")
            outputs = modeld(**inputs)
            last_hidden_states = outputs.last_hidden_state
            tensor1=last_hidden_states[0].sum(0)
            dic={}
            # Compute the cosine similarity between the first and second sentence
            for sentence in sent:
                inpt= tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
                outt = modeld(**inpt)
                lhs= outt.last_hidden_state
                tensor2=lhs
            #     print(tensor1[0].sum(0).shape)
            #     print(tensor2[0].sum(0).shape)
                tensor2=tensor2[0].sum(0)
                similarity = torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=0)
                dic[sentence]=similarity.item()
            tem=sent
            deberta=sorted(tem,key=lambda x:dic[x], reverse=True)
            rankings.append(deberta)
            #monot5
            reranker =  MonoT5()
            query = Query(clickbait[0])
            passages=sent
            texts = [ Text(p, 0) for p in passages]
            reranked = reranker.rerank(query, texts)
            monot5=[]
            for i in range(len(reranked)):
                monot5.append(reranked[i].text)
            rankings.append(monot5)
            #bm25
            copy=sent
            filsen=[]
            dic={}
            for sen in sent:
                sen=remove_pun(sen)
                filsen.append(sen)
    #         print(filsen)
            filsen=tken(filsen)
    #         print(filsen)
            clickbait=clickbait[0]
            cickbait=remove_pun(clickbait)
            tokenized_clickbait=clickbait.split(" ")
            tokenized_clickbait=remove_stopwords(tokenized_clickbait)
            bm25 = BM25Okapi(filsen)
            doc_scores = bm25.get_scores(tokenized_clickbait)
            i=0
            for sen in copy:
                dic[sen]=doc_scores[i]
                i+=1
            tem=sent
            bm=sorted(tem,key=lambda x:dic[x], reverse=True)
            rankings.append(bm)
            def reciprocal(n):
                return 1.0 / n
            diclist=[]
            for rank in rankings:
              tempdic={}
              for i in range(len(rank)):
                tempdic[rank[i]]=i+1
              diclist.append(tempdic)
            fusion_list=[]
            for sen in copy:
              sum=0.0
              for dicidx in diclist:
                num=60+dicidx[sen]
                # print(num,end=' ')
                num=reciprocal(num)
                # print(num)
                sum=sum+num
              fusion_list.append(sum)
            # fused_cols=cols
            new_dic={}
            i=0
            for sen in copy:
                new_dic[sen]=fusion_list[i]
                i+=1
            new_dic
            tem=copy
            sort_fused=sorted(tem,key=lambda x:new_dic[x],reverse=True)
            #Iter-2
            if df2.iloc[index][0]==1:
                ans=[sort_fused[0]]
                answer.append(ans)
            else:
                ans=[]
                lnt=0
                possible=['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty']
                for word in clickbait:
                    for i in range(20):
                        if word==possible[i]:
                            if i+1>lnt:
                                lnt=i+1
                    if word.isdigit():
                        num=int(word)
                        if num<=20:
                            if num>lnt:
                                lnt=num
                if lnt==0:
                    lnt=5
                if len(sort_fused)<lnt:
                    lnt=len(sort_fused)
                for indx in range(lnt):
                    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
                    QA_input = {
                        'question': clickbait,
                        'context': sort_fused[indx]
                    }
                    res = nlp(QA_input)
                    tans=res['answer']
                    if tans=="":
                        tans=sort_fused[indx]
                    ans.append(tans)
                answer.append(ans)




    for i in range(len(df)):
        yield {'uuid': uuids[i], 'spoilerType': [spType(df2.iloc[i][0])], "spoiler": answer[i]}
        # yield {'uuid': uuids[i], 'spoilerType': spType(df2.iloc[i][0])}



def run_baseline(input_file, output_file):
    with open(output_file, 'w') as out:
        for prediction in predict(input_file):
            out.write(json.dumps(prediction) + '\n')


if __name__ == '__main__':
    args = parse_args()
    run_baseline(args.input, args.output)
