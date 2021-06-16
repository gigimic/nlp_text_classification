import json
import pandas as pd
import numpy as np

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

for index, col_n in enumerate(dtf.columns):
    num_uniq = len(dtf[col_n].unique())
    percentage = num_uniq/dtf.shape[0]*100
    print('{0:3d} {1:4d} {2:8.3f}'.format(index, num_uniq, percentage))

## filter categories
# dtf = dtf[ dtf["category"].isin(['ENTERTAINMENT','POLITICS','TECH']) ][["category","headline"]]
## rename columns
# dtf = dtf.rename(columns={"category":"y", "headline":"text"})
## print 5 random rows
dtf.sample(5)

import matplotlib.pyplot as plt
import seaborn as sns

# fig, ax = plt.subplots()
# fig.suptitle("y", fontsize=12)
# dtf["y"].reset_index().groupby("y").count().sort_values(by= 
#        "index").plot(kind="barh", legend=False, 
#         ax=ax).grid(axis='x')
# plt.show()

fig, ax = plt.subplots()
fig.suptitle("category", fontsize=12)
dtf["category"].reset_index().groupby("category").count().sort_values(by= 
       "index").plot(kind="barh", legend=False, 
        ax=ax).grid(axis='x')
plt.show()

