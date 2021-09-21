import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
## for bag-of-words
from sklearn import feature_extraction, feature_selection, model_selection, naive_bayes, pipeline, manifold, preprocessing



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
dtf = dtf[ dtf["category"].isin(['ENTERTAINMENT','TRAVEL','SPORTS']) ][["category","headline"]]
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
# vectorizer = feature_extraction.text.CountVectorizer(max_features=10000, ngram_range=(1,2))

# Tf-Idf (advanced variant of BoW)
# vectorizer = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1,2))


# corpus = dtf_train["text_clean"]
# vectorizer.fit(corpus)
# X_train = vectorizer.transform(corpus)
# dic_vocabulary = vectorizer.vocabulary_

# sns.heatmap(X_train.todense()[:,np.random.randint(0,X_train.shape[1],100)]==0, vmin=0, vmax=1, cbar=False).set_title('Sparse Matrix Sample')
# plt.show()

# this shows the position of the word if it exists in the dictionary
# word = "new york"
# print(dic_vocabulary[word])

# feature selection 
# treat each category as binary (for example, the “Sports” category is 1 for the sports news and 0 for the others);
# perform a Chi-Square test to determine whether a feature and the (binary) target are independent;
# # keep only the features with a certain p-value from the Chi-Square test.

# y = dtf_train["y"]
# X_names = vectorizer.get_feature_names()
# p_value_limit = 0.95
# dtf_features = pd.DataFrame()
# for cat in np.unique(y):
#     chi2, p = feature_selection.chi2(X_train, y==cat)
#     dtf_features = dtf_features.append(pd.DataFrame(
#                    {"feature":X_names, "score":1-p, "y":cat}))
#     dtf_features = dtf_features.sort_values(["y","score"], 
#                     ascending=[True,False])
#     dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
# X_names = dtf_features["feature"].unique().tolist()
# # the number of features are reduced here.


# for cat in np.unique(y):
#    print("# {}:".format(cat))
#    print("  . selected features:",
#          len(dtf_features[dtf_features["y"]==cat]))
#    print("  . top features:", ",".join(dtf_features[dtf_features["y"]==cat]["feature"].values[:10]))
#    print(" ")

# We can refit the vectorizer on the corpus by giving this new set of words as input. 
# That will produce a smaller feature matrix and a shorter vocabulary.

# vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
# vectorizer.fit(corpus)
# X_train = vectorizer.transform(corpus)
# dic_vocabulary = vectorizer.vocabulary_

# This matrix is less sparse 

# Now the model is trained using Naive Bayes 

# classifier = naive_bayes.MultinomialNB()

'''Now build a scikit-learn pipeline: a sequential application of a list of transformations 
 and a final estimator. Putting the Tf-Idf vectorizer and the Naive Bayes classifier 
 in a pipeline allows us to transform and predict test data in just one step.'''
# ## pipeline
# model = pipeline.Pipeline([("vectorizer", vectorizer),  
#                            ("classifier", classifier)])
# ## train classifier
# model["classifier"].fit(X_train, y_train)
# ## test
# # X_test = dtf_test["text_clean"].values
# # predicted = model.predict(X_test)
# # predicted_prob = model.predict_proba(X_test)

# evaluate the performance 
'''Accuracy: the fraction of predictions the model got right.
Confusion Matrix: a summary table that breaks down the number of correct and incorrect predictions by each class.
ROC: a plot that illustrates the true positive rate against the false positive rate 
at various threshold settings. 
The area under the curve (AUC) indicates the probability that the classifier will rank 
a randomly chosen positive observation higher than a randomly chosen negative one.
Precision: the fraction of relevant instances among the retrieved instances.
Recall: the fraction of the total amount of relevant instances that were actually retrieved.'''

classes = np.unique(y_test)
y_test_array = pd.get_dummies(y_test, drop_first=False).values
    
## Accuracy, Precision, Recall
# from sklearn import metrics.accuracy_score
from sklearn import metrics

# accuracy = metrics.accuracy_score(y_test, predicted)
# auc = metrics.roc_auc_score(y_test, predicted_prob, 
#                             multi_class="ovr")
# print("Accuracy:",  round(accuracy,2))
# print("Auc:", round(auc,2))
# print("Detail:")
# print(metrics.classification_report(y_test, predicted))
    
# Plot confusion matrix
# cm = metrics.confusion_matrix(y_test, predicted)
# fig, ax = plt.subplots()
# sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, 
#             cbar=False)
# ax.set_title('Confusion matrix', fontsize=18)
# ax.set_xlabel('Predicted', fontsize=16)
# ax.set_ylabel('True', fontsize=16)
# ax.set(xticklabels=classes, yticklabels=classes)

# # ax.set(xlabel="Predicted", ylabel="True", xticklabels=classes, 
# #        yticklabels=classes, title="Confusion matrix")
# plt.xticks(rotation=0)
# plt.yticks(rotation=60)

# fig, ax = plt.subplots(nrows=1, ncols=2)

# # Plot roc
# for i in range(len(classes)):
#     fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:,i],  
#                            predicted_prob[:,i])
#     ax[0].plot(fpr, tpr, lw=3, 
#               label='{0} (area={1:0.2f})'.format(classes[i], 
#                               metrics.auc(fpr, tpr))
#                )
# ax[0].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
# ax[0].set(xlim=[-0.05,1.0], ylim=[0.0,1.05], 
#           xlabel='False Positive Rate', 
#           ylabel="True Positive Rate (Recall)", 
#           title="Receiver operating characteristic")
# ax[0].legend(loc="lower right")
# ax[0].grid(True)
    
# # Plot precision-recall curve
# for i in range(len(classes)):
#     precision, recall, thresholds = metrics.precision_recall_curve(
#                  y_test_array[:,i], predicted_prob[:,i])
#     ax[1].plot(recall, precision, lw=3, 
#                label='{0} (area={1:0.2f})'.format(classes[i], 
#                                   metrics.auc(recall, precision))
#               )
# ax[1].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall', 
#           ylabel="Precision", title="Precision-Recall curve")
# ax[1].legend(loc="best")
# ax[1].grid(True)
# plt.show()

'''Now we try to understand why the model classifies news with a certain category 
and assess the explainability of these predictions. The lime package can help us 
to build an explainer. To give an illustration, we can take a random observation 
from the test set and see what the model predicts and why.
'''
from lime import lime_text

# ## select observation
# i = 2
# txt_instance = dtf_test["text"].iloc[i]
# ## check true value and predicted value
# print(dtf_test["text"].iloc[i])
# print("True:", y_test[i], "--> Pred:", predicted[i], "| Prob:", round(np.max(predicted_prob[i]),2))
## show explanation
# explainer = lime_text.LimeTextExplainer(class_names=np.unique(y_train))
# explained = explainer.explain_instance(txt_instance, model.predict_proba, num_features=3)
# explained.show_in_notebook(text=txt_instance, predict_proba=False)

'''Word embedding 
In word embedding words of the same context appear together in the corpus
popular models are Word2vec, GloVe and FastText
'''
import gensim
import gensim.downloader as gensim_api

# In Python, a pre-trained Word Embedding model can be loaded from genism-data like this:
# nlp = gensim_api.load("word2vec-google-news-300")

# we can fit our own word2vec on training data corpus woth gensim. 
# The corpus needs to be transformed into a list of list of n-grams.
corpus = dtf_train["text_clean"]
print(corpus.shape)
## create list of lists of unigrams
lst_corpus = []
for string in corpus:
   lst_words = string.split()
   lst_grams = [" ".join(lst_words[i:i+1]) 
               for i in range(0, len(lst_words), 1)]
   lst_corpus.append(lst_grams)

print(len(lst_corpus))
print(lst_corpus[0:10])
## detect bigrams and trigrams
# bigrams_detector = gensim.models.phrases.Phrases(lst_corpus, 
#                  delimiter=" ".encode(), min_count=5, threshold=10)
# bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
# trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus], 
#             delimiter=" ".encode(), min_count=5, threshold=10)
# trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

'''to fit word2vec specify
target size of the word vectors (eg 300)
the window - the max distance between the current and predicted word within a sentence
skip grams sg =1
'''
## fit w2v
# nlp = gensim.models.word2vec.Word2Vec(lst_corpus, vector_size=300,   
#             window=8, min_count=1, sg=1, iter=30)

nlp = gensim.models.word2vec.Word2Vec(lst_corpus, vector_size=300,   
            window=8, min_count=1, sg=1)


## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K
'''
this word embedding is useful in predicting the news category.
this word vectors can be used in neural network as weights 
transform the corpus into padded sequences of word ids to get a feature matrix.
create an embedding matrix so that that the vector of the word with id N is located in the Nth row
build  a neural network with an embedding layer that weighs every word in the sequence with the
corresponding vector
'''
# transform the corpus to a list of sequences
## tokenize text
tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', 
                     oov_token="NaN", 
                     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(lst_corpus)
dic_vocabulary = tokenizer.word_index
## create sequence
lst_text2seq= tokenizer.texts_to_sequences(lst_corpus)
## padding sequence
X_train = kprocessing.sequence.pad_sequences(lst_text2seq, 
                    maxlen=15, padding="post", truncating="post")

print(X_train.shape)
# sns.heatmap(X_train==0, vmin=0, vmax=1, cbar=False)
# plt.show()


i = 0

## list of text: ["I like this", ...]
len_txt = len(dtf_train["text_clean"].iloc[i].split())
print("from: ", dtf_train["text_clean"].iloc[i], "| len:", len_txt)

## sequence of token ids: [[1, 2, 3], ...]
len_tokens = len(X_train[i])
print("to: ", X_train[i], "| len:", len(X_train[i]))

## vocabulary: {"I":1, "like":2, "this":3, ...}
print("check: ", dtf_train["text_clean"].iloc[i].split()[0], 
      " -- idx in vocabulary -->", 
      dic_vocabulary[dtf_train["text_clean"].iloc[i].split()[0]])

print("vocabulary: ", dict(list(dic_vocabulary.items())[0:5]), "... (padding element, 0)")

# Before moving on, don’t forget to do the same feature engineering on the test set as well:

corpus = dtf_test["text_clean"]

## create list of n-grams
lst_corpus = []
for string in corpus:
    lst_words = string.split()
    lst_grams = [" ".join(lst_words[i:i+1]) for i in range(0, 
                 len(lst_words), 1)]
    lst_corpus.append(lst_grams)
    
## detect common bigrams and trigrams using the fitted detectors
# lst_corpus = list(bigrams_detector[lst_corpus])
# lst_corpus = list(trigrams_detector[lst_corpus])
## text to sequence with the fitted tokenizer
lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

## padding sequence
X_test = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=15,
             padding="post", truncating="post")


## start the matrix (length of vocabulary x vector size) with all 0s
embeddings = np.zeros((len(dic_vocabulary)+1, 300))
for word,idx in dic_vocabulary.items():
    ## update the row with vector
    try:
        embeddings[idx] =  nlp[word]
    ## if word not in model then skip and the row stays all 0s
    except:
        pass

word = "data"
print("dic[word]:", dic_vocabulary[word], "|idx")
print("embeddings[idx]:", embeddings[dic_vocabulary[word]].shape, 
      "|vector")

# # Build a deep learning model. The embedding matrix is used in the first 
# Embedding layer of the neural network to classify the news. Each id in the input 
# sequence will be used as the index to access the embedding matrix. The output of this Embedding layer 
# will be a 2D matrix with a word vector for each word id in the input sequence (Sequence length x Vector size). 

## code attention layer
def attention_layer(inputs, neurons):
    x = layers.Permute((2,1))(inputs)
    x = layers.Dense(neurons, activation="softmax")(x)
    x = layers.Permute((2,1), name="attention")(x)
    x = layers.multiply([inputs, x])
    return x

## input
x_in = layers.Input(shape=(15,))
## embedding
x = layers.Embedding(input_dim=embeddings.shape[0],  
                     output_dim=embeddings.shape[1], 
                     weights=[embeddings],
                     input_length=15, trainable=False)(x_in)
## apply attention
x = attention_layer(x, neurons=15)
## 2 layers of bidirectional lstm
x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2, 
                         return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2))(x)
## final dense layers
x = layers.Dense(64, activation='relu')(x)
y_out = layers.Dense(3, activation='softmax')(x)
## compile
model = models.Model(x_in, y_out)
# model = models.Model(x, y_out)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.summary()

## encode y
dic_y_mapping = {n:label for n,label in 
                 enumerate(np.unique(y_train))}
inverse_dic = {v:k for k,v in dic_y_mapping.items()}
y_train = np.array([inverse_dic[y] for y in y_train])
## train
training = model.fit(x=X_train, y=y_train, batch_size=256, 
                     epochs=10, shuffle=True, verbose=0, 
                     validation_split=0.3)
## plot loss and accuracy
metrics_1 = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
ax[0].set(title="Training")
ax11 = ax[0].twinx()
ax[0].plot(training.history['loss'], color='black')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss', color='black')
for metric in metrics_1:
    ax11.plot(training.history[metric], label=metric)
ax11.set_ylabel("Score", color='steelblue')
ax11.legend()
ax[1].set(title="Validation")
ax22 = ax[1].twinx()
ax[1].plot(training.history['val_loss'], color='black')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss', color='black')
for metric in metrics_1:
     ax22.plot(training.history['val_'+metric], label=metric)
ax22.set_ylabel("Score", color="steelblue")
plt.show()
# In some epochs, the accuracy reached 0.89. In order to complete the evaluation of the Word Embedding model, 
# let’s predict the test set and compare the same metrics used before (code for metrics is the same as before).

## test
predicted_prob = model.predict(X_test)
predicted = [dic_y_mapping[np.argmax(pred)] for pred in 
             predicted_prob]



accuracy = metrics.accuracy_score(y_test, predicted)
auc = metrics.roc_auc_score(y_test, predicted_prob, 
                            multi_class="ovr")
print("Accuracy:",  round(accuracy,2))
print("Auc:", round(auc,2))
print("Detail:")
print(metrics.classification_report(y_test, predicted))
    
# # Plot confusion matrix
# cm = metrics.confusion_matrix(y_test, predicted)
# fig, ax = plt.subplots()
# sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, 
#             cbar=False)
# ax.set_title('Confusion matrix', fontsize=18)
# ax.set_xlabel('Predicted', fontsize=16)
# ax.set_ylabel('True', fontsize=16)
# ax.set(xticklabels=classes, yticklabels=classes)

# # ax.set(xlabel="Predicted", ylabel="True", xticklabels=classes, 
# #        yticklabels=classes, title="Confusion matrix")
# plt.xticks(rotation=0)
# plt.yticks(rotation=60)

# fig, ax = plt.subplots(nrows=1, ncols=2)

# # Plot roc
# for i in range(len(classes)):
#     fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:,i],  
#                            predicted_prob[:,i])
#     ax[0].plot(fpr, tpr, lw=3, 
#               label='{0} (area={1:0.2f})'.format(classes[i], 
#                               metrics.auc(fpr, tpr))
#                )
# ax[0].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
# ax[0].set(xlim=[-0.05,1.0], ylim=[0.0,1.05], 
#           xlabel='False Positive Rate', 
#           ylabel="True Positive Rate (Recall)", 
#           title="Receiver operating characteristic")
# ax[0].legend(loc="lower right")
# ax[0].grid(True)
    
# # Plot precision-recall curve
# for i in range(len(classes)):
#     precision, recall, thresholds = metrics.precision_recall_curve(
#                  y_test_array[:,i], predicted_prob[:,i])
#     ax[1].plot(recall, precision, lw=3, 
#                label='{0} (area={1:0.2f})'.format(classes[i], 
#                                   metrics.auc(recall, precision))
#               )
# ax[1].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall', 
#           ylabel="Precision", title="Precision-Recall curve")
# ax[1].legend(loc="best")
# ax[1].grid(True)
# plt.show()
