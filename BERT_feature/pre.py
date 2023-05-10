import pandas as pd

# Read the JSONL file into a pandas DataFrame
df1 = pd.read_json('input.jsonl', lines=True)

click=[]
para=[]
tagID=[]
for ind in df.index:
    click.append(df["postText"][ind])
    para.append(df['targetParagraphs'][ind])
    tagID.append(1) #For test.csv
    
    #For train.csv and dev.csv
    if df['tags'][ind]==['passage']:
        tagID.append(1)
    elif df['tags'][ind]==['phrase']:
        tagID.append(0)
    else:
        tagID.append(2)
        
        
data={'Headline':click,
    'Body':para,
     'label':tagID}
df1=pd.DataFrame(data)
df1.to_csv('test.csv', index=False) #Similarly, make the file for train.csv and dev.csv
