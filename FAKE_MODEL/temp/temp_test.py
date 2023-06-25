import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import warnings, string
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectKBest, chi2


df = pd.read_csv('FAKE_MODEL/Preprocessed Fake Reviews Detection Dataset.csv')

df.dropna(inplace=True)

def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

bow_transformer = CountVectorizer(analyzer=text_process)
bow_transformer.fit(df['text_'])

bow_reviews = bow_transformer.transform(df['text_'])

tfidf_transformer = TfidfTransformer().fit(bow_reviews)

review_train, review_test, label_train, label_test = train_test_split(df['text_'],df['label'],test_size=0.25,random_state=42)


print(review_train.shape)
print(review_test.shape)
print(label_train.shape)
print(label_test.shape)

# pipeline = Pipeline([
#     ('bow', CountVectorizer(analyzer=text_process)),
#     ('feature_selection', SelectKBest(chi2, k=1000)),
#      ('tfidf',TfidfTransformer()),
#     ('classifier', RandomForestClassifier())
# ])

# pipeline.fit(review_train,label_train)

# pickle.dump(pipeline, open('MODEL/naivemodel','wb'))
# model = pickle.load(open('MODEL/naivemodel','rb'))
# predictions = model.predict(review_test)

# print('Classification Report:',classification_report(label_test,predictions))
# print('Confusion Matrix:',confusion_matrix(label_test,predictions))
# print('Accuracy Score:',accuracy_score(label_test,predictions))

# print('/nModel Prediction Accuracy:',str(np.round(accuracy_score(label_test,predictions)*100,2)) + '%')