# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 23:25:48 2020

@author: Asus
"""
import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split

import nltk
#nltk.download('stopwords')

#%matplotlib inline
pd.set_option('display.max_colwidth', 300)



def str_replace(s):
    s = s.replace("[u'", "")
    s = s.replace("']","")
    s = s.replace("u'","")
    s = s.replace("'","")
    s = s.strip()
    return s
    

def str_split(ss):
    result = list(ss.split(','))
    result_new = list(map(str_replace, result))  
    return result_new

def clean_text(text):
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 
    text = re.sub("<BR>", "", text) 
    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    # remove whitespaces 
    text = ' '.join(text.split()) 
    # convert text to lowercase 
    text = text.lower() 
    
    return text


def freq_words(x, terms = 30): 
  all_words = ' '.join([text for text in x]) 
  all_words = all_words.split() 
  fdist = nltk.FreqDist(all_words) 
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())}) 
  
  # selecting top 20 most frequent words 
  d = words_df.nlargest(columns="count", n = terms) 
  
  # visualize words and frequencies
  plt.figure(figsize=(12,15)) 
  ax = sns.barplot(data=d, x= "count", y = "word") 
  ax.set(ylabel = 'Word') 
  plt.show()
  

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)





meta = pd.read_csv("train.csv", sep = ',')
#print(meta.head())
meta.columns = ['id', "movie_id","movie_text","genre"]
movies = meta
print(movies.head())

movies['genre'] = movies['genre'].apply(str_split);
print(movies.head())


#genres_all = [a = sum(s) + a for s in movies['genre']]
all_genres = movies['genre'].sum()
print(len(set(all_genres)))


all_genres = nltk.FreqDist(all_genres) 

# create dataframe
all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()), 
                              'Count': list(all_genres.values())})

g = all_genres_df.nlargest(columns="Count", n = 50) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g, x= "Count", y = "Genre") 
ax.set(ylabel = 'Count') 
plt.show()

movies['clean_plot'] = movies['movie_text'].apply(lambda x: clean_text(x))


# print 100 most frequent words 
freq_words(movies['clean_plot'], 100)


movies['clean_plot'] = movies['clean_plot'].apply(lambda x: remove_stopwords(x))
movies_new = movies


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

movies_new['clean_plot'] = movies_new['clean_plot'].apply(lambda x: remove_stopwords(x))

freq_words(movies_new['clean_plot'], 200)


from sklearn.preprocessing import MultiLabelBinarizer

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(movies_new['genre'])

# transform target variable
y = multilabel_binarizer.transform(movies_new['genre'])

xtrain, xval, ytrain, yval = train_test_split(movies_new['clean_plot'], y, test_size=0.2, random_state=9)

tfidf_vectorizer = TfidfVectorizer(max_df=0.1, max_features=10000)

xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)


from sklearn.linear_model import LogisticRegression

# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier

# Performance metric
from sklearn.metrics import f1_score

lr = LogisticRegression()
clf = OneVsRestClassifier(lr)


# fit model on train data
clf.fit(xtrain_tfidf, ytrain)


# make predictions for validation set
y_pred = clf.predict(xval_tfidf)


f1_score(yval, y_pred, average="micro")

# predict probabilities
y_pred_prob = clf.predict_proba(xval_tfidf)

t = 0.3 # threshold value
y_pred_new = (y_pred_prob >= t).astype(int)

print(f1_score(yval, y_pred_new, average="micro"))


meta_test = pd.read_csv("test.csv", sep = ',')
#print(meta.head())
meta_test.columns = ['id', "movie_text"]
print(meta_test.head())
movies_test = meta_test

movies_test['clean_plot'] = movies_test['movie_text'].apply(lambda x: clean_text(x))


# print 100 most frequent words 
#freq_words(movies['clean_plot'], 100)


movies_test['clean_plot'] = movies_test['clean_plot'].apply(lambda x: remove_stopwords(x))
xtest = movies_test['clean_plot']

xtest_tfidf = tfidf_vectorizer.transform(xtest)


# y_test_prob = clf.predict_proba(xtest_tfidf)

# t = 0.3 # threshold value
# y_pred_test = (y_test_prob >= t).astype(int)

y_pred_test = clf.predict(xtest_tfidf)

genres = multilabel_binarizer.inverse_transform(y_pred_test)

movies_test['genres'] = genres 

movies_test_result = movies_test[['id', 'genres']]

def tuple_to_str(t):
    s = ""
    for i in range(len(t)):
        if (i != len(t) - 1):
            s = s +  str(t[i]) + " "
        else:
            s = s +  str(t[i])
    return(s)    

movies_test_result_str = movies_test_result

movies_test_result_str['genres'] = movies_test_result_str['genres'].apply(tuple_to_str)

movies_test_result_str.to_csv("test_result4.csv", sep = ",", index=None)
#[lambda x : multilabel_binarizer.inverse_transform(s) for s in y_pred_test]

# def infer_tags(q):
#     q = clean_text(q)
#     q = remove_stopwords(q)
#     q_vec = tfidf_vectorizer.transform([q])
#     q_pred = clf.predict(q_vec)
#     return multilabel_binarizer.inverse_transform(q_pred)



# for i in range(5): 
#   k = xval.sample(1).index[0] 
#   print("Movie: ", movies_new['movie_name'][k], "\nPredicted genre: ", infer_tags(xval[k])), print("Actual genre: ",movies_new['genre_new'][k], "\n")

