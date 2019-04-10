#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
import seaborn as sns

data = pd.read_csv('bbc.csv')

data.head()


data['category_id'] = data['type'].factorize()[0]


colslist = ['Index', 'news', 'type', 'category_id']
data.columns = colslist




data.head()


# Before diving head-first into training machine learning models, we should become familiar with the structure and characteristics of our dataset: these properties might inform our problem-solving approach.
# 
# A first step would be to look at some random examples, and the number of examples in each class:



data.groupby('type').Index.count().plot.bar(ylim=0)


data.sample(5, random_state=0)


# We see that the number of articles per class is roughly balanced, which is helpful! If our dataset were imbalanced, we would need to carefully configure our model or artificially balance the dataset, for example by undersampling or oversampling each class.

# # Basic preprocessing

# # Stopwords removal

# Stop words (or commonly occurring words) should be removed from the text data. For this purpose, we can either create a list of stopwords ourselves or we can use predefined libraries.


text_file = open("stopwords.txt", "r")
stopwords = text_file.read().split('\n')




#stopwords = nltk.corpus.stopwords.words('english')




data['news_without_stopwords'] = data['news'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))




print(len(data['news_without_stopwords'][0]))





print(data['news_without_stopwords'])







# # Porter Stemming

# Stemming refers to the removal of suffices, like “ing”, “ly”, “s”, etc. by a simple rule-based approach. For this purpose, we will use PorterStemmer from the NLTK library.


ps = PorterStemmer()





data['news_porter_stemmed'] = data['news_without_stopwords'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))





print(data['news_without_stopwords'][0])



print(data['news_porter_stemmed'][0])


# # Converting to lowercase

# An important pre-processing step is transforming our news articles into lower case. This avoids having multiple copies of the same words. For example, while calculating the word count, ‘Analytics’ and ‘analytics’ will be taken as different words.



data['news_porter_stemmed'] = data['news_porter_stemmed'].apply(lambda x: ' '.join(x.lower() for x in x.split()))





data['news_porter_stemmed'][0]


# # Removing Punctuation
# 

# The next step is to remove punctuation, as it doesn’t add any extra information while treating text data. Therefore removing all instances of it will help us reduce the size of the training data.



data['news_porter_stemmed'] = data['news_porter_stemmed'].str.replace('[^\w\s]','')





data['news_porter_stemmed'][0]


# # Low frequency term filtering (count < 3)

# Remove rarely occurring words from the text. Because they’re so rare, the association between them and other words is dominated by noise.




freq = pd.Series(' '.join(data['news_porter_stemmed']).split()).value_counts()





freq.head()




freq2 = freq[freq <= 3]
freq2





freq3 = list(freq2.index.values)
freq3





data['news_porter_stemmed'] = data['news_porter_stemmed'].apply(lambda x: ' '.join([word for word in x.split() if word not in (freq3)]))




data = data[['Index', 'type', 'category_id', 'news_porter_stemmed']]


# # TF-IDF

# Term frequency is simply the ratio of the count of a word present in a sentence, to the length of the sentence.
# 
# Therefore, we can generalize term frequency as:
# 
# TF = (Number of times term T appears in the particular row) / (number of terms in that row)
# 
# Inverse Document Frequency
# The intuition behind inverse document frequency (IDF) is that a word is not of much use to us if it’s appearing in all the documents.
# 
# Therefore, the IDF of each word is the log of the ratio of the total number of rows to the number of rows in which that word is present.
# 
# IDF = log(N/n), where, N is the total number of rows and n is the number of rows in which the word was present.
# 
# 
# TF-IDF is the multiplication of the TF and IDF 




from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))


# sklearn.feature_extraction.text.TfidfVectorizer will be used to calculate a tf-idf vector for each of our documents.




features = tfidf.fit_transform(data.news_porter_stemmed).toarray()
labels = data.category_id
features.shape


# Each of our 2225 documents is now represented by 15286 features, representing the tf-idf score for different unigrams and bigrams.
# 
# This representation is not only useful for solving our classification task, but also to familiarize ourselves with the dataset. For example, we can use the chi-squared test to find the terms are the most correlated with each of the categories:




data.columns = ['Index', 'newstype', 'category_id', 'news_porter_stemmed']





category_id_df = data[['newstype', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'newstype']].values)




from sklearn.feature_selection import chi2

N = 3
for newstype, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}':".format(newstype))
    print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
    print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))





from sklearn.manifold import TSNE

# Sampling a subset of our dataset because t-SNE is computationally expensive
SAMPLE_SIZE = int(len(features) * 0.3)
np.random.seed(0)
indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE, replace=False)
projected_features = TSNE(n_components=2, random_state=0).fit_transform(features[indices])
colors = ['pink', 'green', 'midnightblue', 'orange', 'darkgrey']




for category, category_id in sorted(category_to_id.items()):
    points = projected_features[(labels[indices] == category_id).values]
    plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[category_id], label=category)
plt.title("tf-idf feature vector for each article, projected on 2 dimensions.",
          fontdict=dict(fontsize=15))
plt.legend()




from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score


models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])



sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,size=8, jitter=True, edgecolor="gray", linewidth=2)




from sklearn.model_selection import train_test_split

model = LogisticRegression(random_state=0)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, data2.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)



from sklearn.metrics import confusion_matrix
import seaborn as sns

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.newstype.values, yticklabels=category_id_df.newstype.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')





model.fit(features, labels)




from sklearn.feature_selection import chi2

N = 5
for newstype, category_id in sorted(category_to_id.items()):
    indices = np.argsort(model.coef_[category_id])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
    bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
    print("# '{}':".format(newstype))
    print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
    print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))





texts = ["Hooli stock price soared after a dip in PiedPiper revenue growth.",
         "Captain Tsubasa scores a magnificent goal for the Japanese team.",
         "Merryweather mercenaries are sent on another mission, as government oversight groups call for new sanctions.",
         "Beyoncé releases a new album, tops the charts in all of south-east Asia!",
         "You won't guess what the latest trend in data analysis is!"]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
    print('"{}"'.format(text))
    print("  - Predicted as: '{}'".format(id_to_category[predicted]))
    print("")





