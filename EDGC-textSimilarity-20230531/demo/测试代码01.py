# -- encoding:utf-8 --

import pandas as pd

df = pd.read_csv("../data/train.csv", sep="\t", header=None, names=['idx', 'text1', 'text2', 'label'])
print(df.head(2))
print(df.text1)
for _, text1, text2, label in df.values:
    print(text1, text2, label)
