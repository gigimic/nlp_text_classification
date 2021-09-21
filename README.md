# Text classification using ML and DL

Multiclass classification  of text with two stategies:
1. Using BOW and TF-IDF and Naive Bayes model
2. Using word embedding (word2vec) and LSTM model

News category dataset is used here. More than 200,000 news articles are included in the data.

### Method 1

Text cleaned.
Corpus created.
stop words removed.
BOW and TFIDF done.
Vocabulary size was kept at 10000.

Three news categories are considered here. ['ENTERTAINMENT','TRAVEL','SPORTS'].

The most common features in each category obtained are:

1. ENTERTAINMENT:
  . selected features: 4566
  . top features: 10,actor,airline,airport,album,amy,around,around world,award,beach
 
2. SPORTS:
  . selected features: 2096
  . top features: 49ers,alabama,allstar,anthem,athlete,ball,baseball,basketball,beat,bowl
 
3. TRAVEL:
  . selected features: 3661
  . top features: 10,abandoned,abroad,adventure,airline,airplane,airport,america,around,around world
 
 Naive Bayes algorithm is used here to do the classification. It is a probabilistic classifier, which makes predictions based on the prior knowledge of conditions that might be related. 

### Method 2

Word embedding was done using gensim word2vec model. 
The length of the text is limited/padded to 15 tokens.
Bidirectional LSTM model is used.
A small improvement in the accuracy was observed.

