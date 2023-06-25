import numpy as np
import pandas as pd
import pickle
import warnings, string
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import time

start_time = time.time()

df = pd.read_csv('FAKE_MODEL/Preprocessed Fake Reviews Detection Dataset.csv')

df.drop('Unnamed: 0',axis=1,inplace=True)
df.dropna(inplace=True)

def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

tfidf_reviews=pickle.load(open('FAKE_MODEL/MODEL/tfidf','rb'))
print("1")

x=df.iloc[:, [3]].values 
y=df.iloc[:, [2]].values 
review_train, review_test, label_train, label_test = train_test_split(x,y,test_size=0.25)
####################################
pipeline1 = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
])

pipeline1.fit(review_train,label_train)
print("5")
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
pickle.dump(pipeline1, open('FAKE_MODEL/MODEL/naivemodel','wb'))

model = pickle.load(open('FAKE_MODEL/MODEL/naivemodel','rb'))
dtree_pred = model.predict(review_test)

print('Classification Report:',classification_report(label_test,dtree_pred))
print('Confusion Matrix:\n',confusion_matrix(label_test,dtree_pred))
print('Accuracy Score:',accuracy_score(label_test,dtree_pred))
print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,dtree_pred)*100,2)) + '%')
##############################################
pipeline2 = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',RandomForestClassifier())
])

pipeline2.fit(review_train,label_train)
print("5")
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
pickle.dump(pipeline2, open('FAKE_MODEL/MODEL/randommodel','wb'))

model = pickle.load(open('FAKE_MODEL/MODEL/randommodel','rb'))
dtree_pred = model.predict(review_test)

print('Classification Report:',classification_report(label_test,dtree_pred))
print('Confusion Matrix:\n',confusion_matrix(label_test,dtree_pred))
print('Accuracy Score:',accuracy_score(label_test,dtree_pred))
print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,dtree_pred)*100,2)) + '%')
##############################################
pipeline3 = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',DecisionTreeClassifier())
])

pipeline3.fit(review_train,label_train)
print("5")
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
pickle.dump(pipeline3, open('FAKE_MODEL/MODEL/decisionmodel','wb'))

model = pickle.load(open('FAKE_MODEL/MODEL/decisionmodel','rb'))
dtree_pred = model.predict(review_test)

print('Classification Report:',classification_report(label_test,dtree_pred))
print('Confusion Matrix:\n',confusion_matrix(label_test,dtree_pred))
print('Accuracy Score:',accuracy_score(label_test,dtree_pred))
print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,dtree_pred)*100,2)) + '%')
##############################################
pipeline4 = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',KNeighborsClassifier(n_neighbors=2))
])

pipeline4.fit(review_train,label_train)
print("5")
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
pickle.dump(pipeline4, open('FAKE_MODEL/MODEL/kneighborsmodel','wb'))

model = pickle.load(open('FAKE_MODEL/MODEL/kneighborsmodel','rb'))
dtree_pred = model.predict(review_test)

print('Classification Report:',classification_report(label_test,dtree_pred))
print('Confusion Matrix:\n',confusion_matrix(label_test,dtree_pred))
print('Accuracy Score:',accuracy_score(label_test,dtree_pred))
print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,dtree_pred)*100,2)) + '%')
##############################################
pipeline5 = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',SVC())
])

pipeline5.fit(review_train,label_train)
print("5")
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
pickle.dump(pipeline5, open('FAKE_MODEL/MODEL/svcmodel','wb'))

model = pickle.load(open('FAKE_MODEL/MODEL/svcmodel','rb'))
dtree_pred = model.predict(review_test)

print('Classification Report:',classification_report(label_test,dtree_pred))
print('Confusion Matrix:\n',confusion_matrix(label_test,dtree_pred))
print('Accuracy Score:',accuracy_score(label_test,dtree_pred))
print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,dtree_pred)*100,2)) + '%')
##############################################
pipeline6 = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',LogisticRegression())
])

pipeline6.fit(review_train,label_train)
print("5")
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
pickle.dump(pipeline6, open('FAKE_MODEL/MODEL/logisticmodel','wb'))

model = pickle.load(open('FAKE_MODEL/MODEL/logisticmodel','rb'))
dtree_pred = model.predict(review_test)

print('Classification Report:',classification_report(label_test,dtree_pred))
print('Confusion Matrix:\n',confusion_matrix(label_test,dtree_pred))
print('Accuracy Score:',accuracy_score(label_test,dtree_pred))
print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,dtree_pred)*100,2)) + '%')
##############################################