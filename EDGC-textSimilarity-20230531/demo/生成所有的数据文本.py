# -- encoding:utf-8 --

import pandas as pd

data_files = ["../data/train.csv", "../data/test.csv", "../data/val.csv"]

with open('../data/sentences.txt', 'w', encoding='utf-8') as writer:
    for data_file in data_files:
        df = pd.read_csv(data_file, sep='\t', header=None)
        for value in df.values:
            writer.writelines(value[1].strip())
            writer.writelines('\n')
            writer.writelines(value[2].strip())
            writer.writelines('\n')

