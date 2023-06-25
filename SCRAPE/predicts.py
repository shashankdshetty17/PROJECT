import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pickle


def preprocess(text):
        return ' '.join([word for word in word_tokenize(str(text)) if word not in stopwords.words('english') and not word.isdigit() and word not in string.punctuation])
    
    
def predictnew(filename,model):
    df=pd.read_csv("SCRAPE/File/"+filename)
    df = df.drop(df.columns[0], axis=1)
    if 'prediction' in df.columns:
        df = df.drop('pred', axis=1)
    
    df['review_text'].dropna(inplace=True)
    df['s']=df['review_text']+" "+df['review_title']
    print(model)
    
    df['text'] = df['s'].apply(preprocess)
    
    df['text'] = df['text'].str.lower()
    
    stemmer = PorterStemmer()
    def stem_words(text):
        return ' '.join([stemmer.stem(word) for word in text.split()])
    df['text'] = df['text'].apply(lambda x: stem_words(x))
    
    lemmatizer = WordNetLemmatizer()
    def lemmatize_words(text):
        return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    df["text"] = df["text"].apply(lambda text: lemmatize_words(text))
    if 'result' in df.columns:
        df = df.drop('result', axis=1)
    if(model==1):
        predict1(df["text"],df,filename)
    elif(model==2):
        predict1(df["text"],df,filename)
        predict2(df["text"],df,filename)
    elif(model==3):
        predict1(df["text"],df,filename)
        predict2(df["text"],df,filename)
        predict3(df["text"],df,filename)
    elif(model==4):
        predict1(df["text"],df,filename)
        predict2(df["text"],df,filename)
        predict3(df["text"],df,filename)
        predict4(df["text"],df,filename)
    elif(model==5):
        predict1(df["text"],df,filename)
        predict2(df["text"],df,filename)
        predict3(df["text"],df,filename)
        predict4(df["text"],df,filename)
        predict5(df["text"],df,filename)
    
    last_four_columns = df.iloc[:, -(model):] 
    df['Result'] = last_four_columns.apply(lambda x: x.value_counts().idxmax(), axis=1)
    print(df)
    return df
    
    
def predict1(r,df,filename):
      
    model = pickle.load(open('FAKE_MODEL/MODEL/svcmodel','rb'))
    predictions = model.predict(r)
    count_or = np.count_nonzero(predictions == 'OR')
    count_cg = np.count_nonzero(predictions == 'CG')
    print(predictions,count_cg,count_or)
    if 'svc' in df.columns:
        df = df.drop('svc', axis=1)
    df['svc']=predictions
    df = df.drop(['s', 'text'], axis=1)
    df.to_csv("SCRAPE/File/"+filename)
    
def predict2(r,df,filename):   
    model = pickle.load(open('FAKE_MODEL/MODEL/logisticmodel','rb'))
    predictions = model.predict(r)
    count_or = np.count_nonzero(predictions == 'OR')
    count_cg = np.count_nonzero(predictions == 'CG')
    print(predictions,count_cg,count_or)
    if 'logistic' in df.columns:
        df = df.drop('logistic', axis=1)
    df['logistic']=predictions
    df = df.drop(['s', 'text'], axis=1)
    df.to_csv("SCRAPE/File/"+filename)
def predict3(r,df,filename):
      
    model = pickle.load(open('FAKE_MODEL/MODEL/naivemodel','rb'))
    predictions = model.predict(r)
    count_or = np.count_nonzero(predictions == 'OR')
    count_cg = np.count_nonzero(predictions == 'CG')
    print(predictions,count_cg,count_or)
    if 'naive' in df.columns:
        df = df.drop('naive', axis=1)
    df['naive']=predictions
    df = df.drop(['s', 'text'], axis=1)
    df.to_csv("SCRAPE/File/"+filename)
def predict4(r,df,filename):
      
    model = pickle.load(open('FAKE_MODEL/MODEL/randommodel','rb'))
    predictions = model.predict(r)
    count_or = np.count_nonzero(predictions == 'OR')
    count_cg = np.count_nonzero(predictions == 'CG')
    print(predictions,count_cg,count_or)
    if 'random' in df.columns:
        df = df.drop('random', axis=1)
    df['random']=predictions
    df = df.drop(['s', 'text'], axis=1)
    df.to_csv("SCRAPE/File/"+filename)
def predict5(r,df,filename):
      
    model = pickle.load(open('FAKE_MODEL/MODEL/decisionmodel','rb'))
    predictions = model.predict(r)
    count_or = np.count_nonzero(predictions == 'OR')
    count_cg = np.count_nonzero(predictions == 'CG')
    print(predictions,count_cg,count_or)
    if 'decision' in df.columns:
        df = df.drop('decision', axis=1)
    df['decision']=predictions
    df = df.drop(['s', 'text'], axis=1)
    df.to_csv("SCRAPE/File/"+filename)
def predict6(r,df,filename):
      
    model = pickle.load(open('FAKE_MODEL/MODEL/kneighborsmodel','rb'))
    predictions = model.predict(r)
    count_or = np.count_nonzero(predictions == 'OR')
    count_cg = np.count_nonzero(predictions == 'CG')
    print(predictions,count_cg,count_or)
    if 'kneighborsmodel' in df.columns:
        df = df.drop('kneighborsmodel', axis=1)
    df['kneighborsmodel']=predictions
    df = df.drop(['s', 'text'], axis=1)
    df.to_csv("SCRAPE/File/"+filename)

