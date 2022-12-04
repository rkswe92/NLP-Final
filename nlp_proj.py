# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 16:11:43 2022

@author: ravik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as ply
import nltk
import torch
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
special_characters = ['!','#','$','%', '&','@','[',']',' ',']','_']

def findmaxofsentiment(dict_):
    result=''
    if (dict_['neu'] > dict_['neg']) and (dict_['neu'] > dict_['pos']):
        result = 'neutral sentiment'
    elif dict_['pos'] > dict_['neg']:
        result ='positive sentiment'
    elif dict_['neg'] > dict_['pos']:
        result = 'negetive sentiment'
    return result

def moviePositveNegetiveNeutralIndex(df,movie_col,result_col):
    movies = df[movie_col].unique().tolist()
    mov=[]
    p_index=[]
    neu_index=[]
    neg_index=[]
    for m in movies:
        mov.append(m.replace('\t',''))
        count=0
        positive=0
        neutral=0
        negative=0
        for i,r in df.iterrows():
            if m == r[movie_col] :
                if r[result_col] =='positive sentiment':
                    positive = positive + 1
                elif r[result_col] =='neutral sentiment':
                    neutral= neutral + 1
                else:
                    negative=negative + 1           
                count= count + 1
        p_index.append(positive/count)
        neu_index.append(neutral/count)
        neg_index.append(negative/count)
    return pd.DataFrame((list(zip(mov, p_index,neu_index,neg_index))),
               columns = ['Movie', 'P_index','neu_index','neg_index'])
          
        
    

#removeing special characters and numbers
def removeSpecialCharacters(value):
    for i in special_characters:
        return value.replace(i,'')
    
#removing numbers    
def removeNumbers(df, col):   
       return df[col].str.extract('(\w+\s\w+[^0-9]|\w+[^0-9])')

#Lowercase Fucntion
def covertToLower(df,col):
    return list(map(lambda x: x.lower(), df[col]))

def wordTokenizerfunc(df,col):
    return df[col].apply(word_tokenize)

def removeStopWords(df,col):
    return df[col].apply(lambda x: [item for item in x if item not in stop])

def sentiment_score(review):
    tokens = tokenizer.encode(review,add_special_tokens=False)
    result = model(tokens)
    return int(torch.argmax(result.logits))+1

def assignPosNeuNegValuestoColumns(df,col):
    sen=[]
    pos=[]
    neu=[]
    neg=[]
    sia = SentimentIntensityAnalyzer()
    for i in df[col]:
        st= sia.polarity_scores(' '.join(i))
        pos.append(st['pos'])
        neu.append(st['neu'])
        neg.append(st['neg'])
        sen.append(sia.polarity_scores(' '.join(i)))
    df['sen'] =  sen
    df['pos'] =  pos
    df['neu'] =  neu
    df['neg'] =  neg
    return df
print('IMDB')
df_imdb = pd.read_excel('IMDB.xlsx', 'IMDB')
df_imdb['Reviews'] = covertToLower(df_imdb,'Reviews')

#removing Numbers
#df_imdb['Reviews'] = removeNumbers(df_imdb,'Reviews')

#Tokenizing the reviews
df_imdb['tokenized_text'] = wordTokenizerfunc(df_imdb,'Reviews')
        
#Applying the stopwords
df_imdb['tokenized_text'] = removeStopWords(df_imdb,'tokenized_text')


#Applying NLTK Sentimentanlysis
df_imdb=assignPosNeuNegValuestoColumns(df_imdb,'tokenized_text')
df_imdb['result']=df_imdb['sen'].apply(findmaxofsentiment)
final_df_imdb = moviePositveNegetiveNeutralIndex(df_imdb,'Movie Name','result')
print(final_df_imdb.head(10))
# =============================================================================
print('Rotten')
df_rotten = pd.read_excel('Rotten.xlsx','Rotten')
df_rotten['Reviews'] = covertToLower(df_rotten,'Reviews')
 #removing Numbers
#df_imdb['Reviews'] = removeNumbers(df_imdb,'Reviews')

#Tokenizing the reviews
df_rotten['tokenized_text'] = wordTokenizerfunc(df_rotten,'Reviews')
         
 #Applying the stopwords
df_rotten['tokenized_text'] = removeStopWords(df_rotten,'tokenized_text')
#Applying NLTK Sentimentanlysis
df_rotten = assignPosNeuNegValuestoColumns(df_rotten,'tokenized_text')
df_rotten['result']=df_rotten['sen'].apply(findmaxofsentiment)
final_df_rotten = moviePositveNegetiveNeutralIndex(df_rotten,'Movie Name','result')
print(final_df_rotten.head(10))
# =============================================================================
