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
fig, ax = plt.subplots()
fig.suptitle("category", fontsize=12)
dtf["category"].reset_index().groupby("category").count().sort_values(by= 
       "index").plot(kind="barh", legend=False, 
        ax=ax).grid(axis='x')
plt.show()
