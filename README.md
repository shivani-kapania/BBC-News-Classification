# BBC-News-Classification



## Dataset Description: ##

Consists of 2225 documents from the BBC news website corresponding to stories in five topical areas from 2004-2005.

Class Labels: 5 (business, entertainment, politics, sport, tech)

[Dataset](http://mlg.ucd.ie/datasets/bbc.html)


## Method ##

* Combined the data .txt files to a csv with columns news article and news type. 
* Applied pre-processing steps like stopwords removal, porter stemming, conversion to lowercase etc. 
* Extracted features using TF-IDF
* Divided the feature extracted dataset into two parts train and test set and trained Logistic Regression on it. 
