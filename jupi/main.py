import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

pd.options.mode.chained_assignment = None


def load_data():
    df=pd.read_csv("Fake-Review-Detection-master/Data/fake reviews dataset.csv")
    print("reading done")
    print(df.size)
    return df


def data_cleaning(df):
    
    # Pre-processing Text Reviews
    # Remove Stop Words
    stop = stopwords.words('english')
    df['text_'] = df['text_'].apply(
        lambda x: ' '.join(word for word in str(x).split() if word not in stop))

    # Remove Punctuations
    tokenizer = RegexpTokenizer(r'\w+')
    df['text_'] = df['text_'].apply(
        lambda x: ' '.join(word for word in tokenizer.tokenize(x)))

    # Lowercase Words
    df['text_'] = df['text_'].apply(
        lambda x: str(x).lower())
    
    df.drop([ 'category'], axis=1, inplace=True)
  
    print("Data Cleaning Complete")
    return df

    

def feature_engineering(df):
    print("Feature Engineering: Creating New Features")
    
    # Review Length
    df['rl'] = df['text_'].apply(
        lambda x: len(x.split()))
    df.drop(index=np.where(pd.isnull(df))[0], axis=0, inplace=True)

    sid = SentimentIntensityAnalyzer()
    for index, row in df.iterrows():
        text1 = row['text_']
        sentiment_dict = sid.polarity_scores(text1)
        df.at[index, 'sentiment'] =sentiment_dict['compound']
        
    ########
    def text_process(review):
        nopunc = [char for char in review if char not in string.punctuation]
        nopunc = ''.join(nopunc)
        return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    
    bow_transformer = CountVectorizer(analyzer=text_process)


    bow_transformer.fit(df['text_'])
    print("Total Vocabulary:",len(bow_transformer.vocabulary_))

    bow_reviews = bow_transformer.transform(df['text_'])

    tfidf_transformer = TfidfTransformer().fit(bow_reviews)
   
    tfidf_reviews = tfidf_transformer.transform(bow_reviews)
    df['tfidf'] = pd.Series(tfidf_reviews.toarray().tolist())
    ########
    
    print("Feature Engineering Complete")
    return df

    


def under_sampling(df):
    print("Under-Sampling Data")
    # Count of Reviews
    # print("Authentic", len(df[(df['flagged'] == 'N')]))
    # print("Fake", len(df[(df['flagged'] == 'Y')]))

    sample_size = len(df[(df['label'] == 'OR')])

    authentic_reviews_df = df[df['label'] == 'OR']
    fake_reviews_df = df[df['label'] == 'CG']

    authentic_reviews_us_df = authentic_reviews_df.sample(sample_size)
    under_sampled_df = pd.concat([authentic_reviews_us_df, fake_reviews_df], axis=0)

    print("Under-Sampling Complete")
    return under_sampled_df


def semi_supervised_learning(df, model, algorithm, threshold=0.8, iterations=40):
    df = df.copy()
    print("Training "+algorithm+" Model")
    labels = df['label']

    df.drop([ 'label','text_'], axis=1, inplace=True)

    train_data, test_data, train_label, test_label = train_test_split(df, labels, test_size=0.3, random_state=40)

    test_data_copy = test_data.copy()
    test_label_copy = test_label.copy()

    all_labeled = False

    current_iteration = 0

    pbar = tqdm(total=iterations)

    while not all_labeled and (current_iteration < iterations):
        # print("Before train data length : ", len(train_data))
        # print("Before test data length : ", len(test_data))
        current_iteration += 1
        model.fit(train_data, train_label)

        probabilities = model.predict_proba(test_data)
        pseudo_labels = model.predict(test_data)

        indices = np.argwhere(probabilities > threshold)

        # print("rows above threshold : ", len(indices))
        for item in indices:
            train_data.loc[test_data.index[item[0]]] = test_data.iloc[item[0]]
            train_label.loc[test_data.index[item[0]]] = pseudo_labels[item[0]]
        test_data.drop(test_data.index[indices[:, 0]], inplace=True)
        test_label.drop(test_label.index[indices[:, 0]], inplace=True)
        # print("After train data length : ", len(train_data))
        # print("After test data length : ", len(test_data))
        print("--" * 20)

        if len(test_data) == 0:
            print("Exiting loop")
            all_labeled = True
        pbar.update(1)
    pbar.close()
    predicted_labels = model.predict(test_data_copy)

    # print('Best Params : ', grid_clf_acc.best_params_)
    print(algorithm + ' Model Results')
    print('--' * 20)
    print('Accuracy Score : ' + str(accuracy_score(test_label_copy, predicted_labels)))
    print('Precision Score : ' + str(precision_score(test_label_copy, predicted_labels, pos_label="CG")))
    print('Recall Score : ' + str(recall_score(test_label_copy, predicted_labels, pos_label="CG")))
    print('F1 Score : ' + str(f1_score(test_label_copy, predicted_labels, pos_label="CG")))
    print('Confusion Matrix : \n' + str(confusion_matrix(test_label_copy, predicted_labels)))
    plot_confusion_matrix(test_label_copy, predicted_labels, classes=['OR', 'CG'],title=algorithm + ' Confusion Matrix').show()


def plot_confusion_matrix(y_true, y_pred, classes, title=None, cmap=plt.cm.Blues):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return plt


def main():
    start_time = time()
    df = load_data()
    print(df)
    df = data_cleaning(df)
    print(df)
    df = feature_engineering(df)
    #under_sampled_df = df
    under_sampled_df = under_sampling(df)
    rf = RandomForestClassifier(random_state=42, criterion='entropy', max_depth=14, max_features='sqrt',n_estimators=500)
    nb = GaussianNB()
    print(df)
    semi_supervised_learning(under_sampled_df, model=rf, threshold=0.8, iterations=5, algorithm='Random Forest')
    semi_supervised_learning(under_sampled_df, model=nb, threshold=0.8, iterations=5, algorithm='Naive Bayes')
    end_time = time()
    print("Time taken : ", end_time - start_time)


if __name__ == '__main__':
    main()
