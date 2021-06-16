import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
## for bag-of-words
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing


from utils_fns import utils_preprocess_text

lst_dics = []
with open('News_Category_Dataset_v2.json', mode='r', errors='ignore') as json_file:
    for dic in json_file:
        lst_dics.append( json.loads(dic) )

# print(lst_dics[0])
# print(lst_dics[1])

dtf = pd.DataFrame(lst_dics)
print(dtf.shape)
print(dtf.describe())
print(dtf.head())

print(dtf['category'].unique())
print(dtf.nunique())

# ploltting the categories - total 41 categories 
# fig, ax = plt.subplots()
# fig.suptitle("category", fontsize=12)
# dtf["category"].reset_index().groupby("category").count().sort_values(by= 
#        "index").plot(kind="barh", legend=False, 
#         ax=ax).grid(axis='x')
# plt.show()


for index, col_n in enumerate(dtf.columns):
    num_uniq = len(dtf[col_n].unique())
    percentage = num_uniq/dtf.shape[0]*100
    print('{0:3d} {1:4d} {2:8.3f}'.format(index, num_uniq, percentage))

## filter categories - keeping only 3 categories
dtf = dtf[ dtf["category"].isin(['ENTERTAINMENT','POLITICS','SPORTS']) ][["category","headline"]]
## rename columns
dtf = dtf.rename(columns={"category":"y", "headline":"text"})
## print 5 random rows
print(dtf.sample(5))


# plotting the selected categories

# fig, ax = plt.subplots()
# fig.suptitle("y", fontsize=12)
# dtf["y"].reset_index().groupby("y").count().sort_values(by= 
#        "index").plot(kind="barh", legend=False, 
#         ax=ax).grid(axis='x')
# plt.show()

lst_stopwords = nltk.corpus.stopwords.words("english")

dtf["text_clean"] = dtf["text"].apply(lambda x: 
          utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, 
          lst_stopwords=lst_stopwords))
# print(dtf.head())

# split dataset
dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.3)
## get target
y_train = dtf_train["y"].values
y_test = dtf_test["y"].values

## Count (classic BoW)
vectorizer = feature_extraction.text.CountVectorizer(max_features=10000, ngram_range=(1,2))

corpus = dtf_train["text_clean"]
vectorizer.fit(corpus)
X_train = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_

# sns.heatmap(X_train.todense()[:,np.random.randint(0,X_train.shape[1],100)]==0, vmin=0, vmax=1, cbar=False).set_title('Sparse Matrix Sample')
# plt.show()

# this shows the position of the word if it exists in the dictionary
word = "new york"
print(dic_vocabulary[word])
