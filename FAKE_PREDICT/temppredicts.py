import pandas as pd
import numpy as np
import string, nltk
import joblib

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pickle
def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def preprocess(text):
        return ' '.join([word for word in word_tokenize(str(text)) if word not in stopwords.words('english') and not word.isdigit() and word not in string.punctuation])
    
    
def predictnew(filename):
    df=pd.read_csv("SCRAPE/File/"+filename)
    df['review_text'].dropna(inplace=True)
    df['s']=df['review_text']+" "+df['review_title']
    print(df['s'])
    
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
    predict1(df["text"],df,filename)
    
def predict1(r,df,filename):
    model = pickle.load(open('FAKE_MODEL/MODEL/svcmodel','rb'))
    predictions = model.predict(r)
    count_or = np.count_nonzero(predictions == 'OR')
    count_cg = np.count_nonzero(predictions == 'CG')
    print(predictions,count_cg,count_or)
    #df['prediction']=predictions
    #df.to_csv("SCRAPE/File/"+filename)
    
    
predictnew("Tusmad_reviews.csv")
