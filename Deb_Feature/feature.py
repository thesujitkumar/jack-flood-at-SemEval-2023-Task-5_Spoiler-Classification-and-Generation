import pandas as pd

# Read the JSONL file into a pandas DataFrame
df = pd.read_json('input.jsonl', lines=True)
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from nltk.tokenize import sent_tokenize
import re
def remove_stopwords(words):
    return [word for word in words if word not in stop_words]
def remove_pun(text):
    res= re.sub(r'[-"()\#@.,‘:?!]', ' ', text)
    # remove additional space from string
    res = re.sub(' +', ' ', res)
    #res = res.lower()
    return res
from string import punctuation
from collections import Counter
idx=0
data=[]
for ind in df.index:
    click=df["postText"][ind]
    para=df["targetParagraphs"][ind]
    sent=[]
    wordc=0
    parac=len(para)
    for sentences in para:
            temp=sent_tokenize(sentences)
            for tempo in temp:
                sent.append(tempo)
    sentc=len(sent)
    c = Counter(c for line in sent for c in line if c in punctuation)
#     print(c)
    ans=''
    for s in sent:
        ans+=s
    ans=remove_pun(ans)
    tokenized_ans=ans.split(" ")
    tokenized_ans=remove_stopwords(tokenized_ans)
#     print(parac)
#     print(sentc)
    punc=0
#     print(tokenized_ans)
    wordc=len(tokenized_ans)
    dict={}
    for word in tokenized_ans:
        if word not in dict:
            dict[word]=0
        else:
            dict[word]+=1
    for key in c:
        punc+=c[key]
#     print(punc)
    cli=[]
    for sentences in click:
            temp=sent_tokenize(sentences)
            for tempo in temp:
                cli.append(tempo)
    clisc=len(cli)
    c = Counter(c for line in cli for c in line if c in punctuation)
#     print(c)
    ans=''
    for s in cli:
        ans+=s
    ans=remove_pun(ans)
    tokenized_ans=ans.split(" ")
    tokenized_ans=remove_stopwords(tokenized_ans)
    clipc=0
    cliwc=len(tokenized_ans)
    count=0
    for word in tokenized_ans:
        if word in dict:
            count+=dict[word]
    for key in c:
        clipc+=c[key]
    vec=[]
    vec.append(wordc)
    vec.append(sentc)
    vec.append(punc)
    vec.append(parac)
    vec.append(cliwc)
    vec.append(clisc)
    vec.append(clipc)
    vec.append(count)
    data.append(vec)
df1=pd.DataFrame(data)
