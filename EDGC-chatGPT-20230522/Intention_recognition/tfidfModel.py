#!/usr/bin/env python3
# coding: utf-8

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from typing import List
import json
import jieba

class loadData:
    def __init__(self):

        self.data = '../query_data/query_data.txt'
        self.listData = []
        self.jiebaDict = './jieba_equipment.txt'
    def load(self):

        jieba.load_userdict(self.jiebaDict)
        with open(self.data,'r',encoding='utf-8') as r:
            allData = r.readlines()
            for i in allData:
                i = i.strip()
                i = json.loads(i)
                words = jieba.cut(i["text"])

                self.listData.extend(words)

        return self.listData

class tfidfModel():
    def __init__(self):
        self.data = loadData().load()

        self.jiebaDict = './jieba_equipment.txt'

    def train_model(self):
        # create TfidfVectorizer object
        vector = TfidfVectorizer()

        #load data....  data--> list
        vectors = vector.fit(self.data)

        joblib.dump(vector,'./tfidf_model.pkl')

        return vectors

    def test(self,new_data):
        # 加载模型
        vectorizer = joblib.load('./tfidf_model.pkl')
        # jieba.load_userdict(self.jiebaDict)
        # newData = jieba.lcut(new_data)
        # print("newData",newData)
        # 将新的文本向量化
        # new_vectors = vectorizer.transform(newData).toarray()
        new_vectors = vectorizer.transform(new_data).toarray()

        print("new_vectors:",new_vectors)













if __name__=="__main__":
    model = tfidfModel()
    # model.train_model()
    # model.test('MZX004A型1000M信道密码机的主要功能是什么')
    model.test(['MZX004A型1000M信道密码机的主要功能是什么'])

    # data = loadData()
    # a = data.load()
    # print(a)






















