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

# Tf-Idf (advanced variant of BoW)
# vectorizer = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1,2))


corpus = dtf_train["text_clean"]
vectorizer.fit(corpus)
X_train = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_

# sns.heatmap(X_train.todense()[:,np.random.randint(0,X_train.shape[1],100)]==0, vmin=0, vmax=1, cbar=False).set_title('Sparse Matrix Sample')
# plt.show()

# this shows the position of the word if it exists in the dictionary
word = "new york"
print(dic_vocabulary[word])

# feature selection 
# treat each category as binary (for example, the “Sports” category is 1 for the sports news and 0 for the others);
# perform a Chi-Square test to determine whether a feature and the (binary) target are independent;
# keep only the features with a certain p-value from the Chi-Square test.

y = dtf_train["y"]
X_names = vectorizer.get_feature_names()
p_value_limit = 0.95
dtf_features = pd.DataFrame()
for cat in np.unique(y):
    chi2, p = feature_selection.chi2(X_train, y==cat)
    dtf_features = dtf_features.append(pd.DataFrame(
                   {"feature":X_names, "score":1-p, "y":cat}))
    dtf_features = dtf_features.sort_values(["y","score"], 
                    ascending=[True,False])
    dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
X_names = dtf_features["feature"].unique().tolist()
# the number of features are reduced here.


for cat in np.unique(y):
   print("# {}:".format(cat))
   print("  . selected features:",
         len(dtf_features[dtf_features["y"]==cat]))
   print("  . top features:", ",".join(dtf_features[dtf_features["y"]==cat]["feature"].values[:10]))
   print(" ")

# We can refit the vectorizer on the corpus by giving this new set of words as input. 
# That will produce a smaller feature matrix and a shorter vocabulary.

vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
vectorizer.fit(corpus)
X_train = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_

# This matrix is less sparse 

# Now the model is trained using Naive Bayes 

classifier = naive_bayes.MultinomialNB()

'''Now build a scikit-learn pipeline: a sequential application of a list of transformations 
 and a final estimator. Putting the Tf-Idf vectorizer and the Naive Bayes classifier 
 in a pipeline allows us to transform and predict test data in just one step.'''
## pipeline
model = pipeline.Pipeline([("vectorizer", vectorizer),  
                           ("classifier", classifier)])
## train classifier
model["classifier"].fit(X_train, y_train)
## test
X_test = dtf_test["text_clean"].values
predicted = model.predict(X_test)
predicted_prob = model.predict_proba(X_test)

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

accuracy = metrics.accuracy_score(y_test, predicted)
auc = metrics.roc_auc_score(y_test, predicted_prob, 
                            multi_class="ovr")
print("Accuracy:",  round(accuracy,2))
print("Auc:", round(auc,2))
print("Detail:")
print(metrics.classification_report(y_test, predicted))
    
## Plot confusion matrix
# cm = metrics.confusion_matrix(y_test, predicted)
# fig, ax = plt.subplots()
# sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, 
#             cbar=False)
# ax.set(xlabel="Pred", ylabel="True", xticklabels=classes, 
#        yticklabels=classes, title="Confusion matrix")
# plt.yticks(rotation=0)

# fig, ax = plt.subplots(nrows=1, ncols=2)

## Plot roc
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
    
## Plot precision-recall curve
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

## select observation
i = 0
txt_instance = dtf_test["text"].iloc[i]
## check true value and predicted value
print("True:", y_test[i], "--> Pred:", predicted[i], "| Prob:", round(np.max(predicted_prob[i]),2))
## show explanation
explainer = lime_text.LimeTextExplainer(class_names=np.unique(y_train))
explained = explainer.explain_instance(txt_instance, model.predict_proba, num_features=3)
explained.show_in_notebook(text=txt_instance, predict_proba=False)

