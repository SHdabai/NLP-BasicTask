# -*- coding: UTF-8 -*-
from pprint import pprint
import numpy as np
from functools import reduce
import jieba
import json
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB



class loadData():
    def __init__(self):

        self.data = '../query_data/query_data.txt'
        self.listData = []
        self.jiebaDict = './jieba_equipment.txt'
        self.stopwords_path = './stop_words.utf8'
        self.stopwords = [w.strip() for w in open(self.stopwords_path, 'r', encoding='utf8') if w.strip()]
        self.tfidfModel = './tfidf_model.pkl'

        self.fault_cause_qwds = ['什么原因', '什么引起', '引起', '导致', '什么因素', '哪些因素', '怎么引起',
                                 '怎么导致', '为啥出现', '为什么会出现', '出现', '原因是什么', '原因']  # 查询故障原因
        self.function_qwds = ['什么功能', '哪些功能', '可以干什么', '如何应用', '功能是什么', '做什么', '做啥', '干啥', '啥用',
                              '什么用', '能干啥', '能做啥']  # 查询功能
        self.use_qwds = ['怎么用', '如何使用', '怎么使用', '咋用', '咋使用', '如何操作', '咋操作', '使用',
                         '操作', '用', '操作步骤', '使用流程', '使用说明']  # 查询操作
        self.composition_qwds = ['组成部分', '组成成分', '什么构成', '构成', '组成', '包含',
                                 '结构', '内容', '哪些零件', '零件', '构件', '组建']  # 查询组成
        self.solution_qwds = ['怎么解决', '咋解决', '解决', '如何处理', '处理', '修复', '怎么办',
                              '咋弄', '咋整', '咋办', '怎么弄', '怎么整']  # 查询解决方案
        self.configuration_qwds = ['配置信息', '配置', '配置内容', '配置啥', '配置什么', '哪些配置', '怎么配置']  # 查询配置

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

    def features(self, text):
        """
        提取问题的TF-IDF特征
        :param text:
        :param vectorizer:
        :return:
        """
        jieba.load_userdict(self.jiebaDict)
        words = [w.strip() for w in jieba.cut(text) if w.strip() and w.strip() not in self.stopwords]
        sents = [' '.join(words)]
        vec = joblib.load('./tfidf_model.pkl')
        tfidf = vec.transform(sents).toarray()

        other_feature = self.other_features(text)  # 这个是根据模板匹配的目的特征
        m = other_feature.shape  # 查看数据维度
        other_feature = np.reshape(other_feature, (1, m[0]))  # 数据特征广播

        feature = np.concatenate((tfidf, other_feature), axis=1)  # 两者特征进行拼接

        return feature
        # print("feature:::",feature)


    def other_features(self, text):
        """
        提取问题的关键词特征
        :param text:
        :return:
        """

        features = [0] * 7
        for d in self.fault_cause_qwds:
            if d in text:
                features[0] += 1

        for s in self.function_qwds:
            if s in text:
                features[1] += 1

        for c in self.use_qwds:
            if c in text:
                features[2] += 1

        for c in self.composition_qwds:
            if c in text:
                features[3] += 1
        for p in self.solution_qwds:
            if p in text:
                features[4] += 1

        for r in self.configuration_qwds:
            if r in text:
                features[5] += 1

        m = max(features)
        n = min(features)
        normed_features = []
        if m == n:
            normed_features = features
        else:
            for i in features:
                j = (i - n) / (m - n)
                normed_features.append(j)

        return np.array(normed_features)



class Bayes():
    def __init__(self):
        self.data = '../query_data/query_data.txt'
        self.listData = []
        self.labelData = []
        self.jiebaDict = './jieba_equipment.txt'
        self.stopwords_path = './stop_words.utf8'
        self.stopwords = [w.strip() for w in open(self.stopwords_path, 'r', encoding='utf8') if w.strip()]
        self.loadData = loadData()
        self.idtolabel = {'0': 'query_solution', '1': 'query_fault_cause', '2': 'query_configuration',
                     '3': 'query_use', '4': 'query_composition', '5': 'query_parameter', '6': 'query_function',
                     '7': 'query_operation', }

        self.labeltoid = {'query_solution': '0', 'query_fault_cause': '1', 'query_configuration': '2',
 'query_use': '3', 'query_composition': '4', 'query_parameter': '5',
 'query_function': '6', 'query_operation': '7'}

    def train_data(self):


        jieba.load_userdict(self.jiebaDict)
        with open(self.data,'r',encoding='utf-8') as r:
            allData = r.readlines()
            for i in allData:
                i = i.strip()
                i = json.loads(i)
                query_data = i['text']
                query_data_feature = self.loadData.features(query_data)

                self.listData.append(query_data_feature)
                if i["label"] in self.labeltoid:
                    label = self.labeltoid[i["label"]]

                    self.labelData.append(label)




            flattened_train_data = np.concatenate([arr.reshape(arr.shape[0], -1) for arr in self.listData])

            X_train = flattened_train_data

            y_train = self.labelData


        return X_train,y_train
        # print(X_train)
        # print(len(X_train))
        # print(y_train)
        # print(len(y_train))
        # print(flattened_train_data)
        # print(len(flattened_train_data))
    #
    def train(self):
        # 创建模型
        model = MultinomialNB()

        X_train,y_train = self.train_data()
        flattened_train_data = np.concatenate([arr.reshape(arr.shape[0], -1) for arr in X_train])

        # 进行训练
        model.fit(X_train, y_train)

        joblib.dump(model, 'model.pkl')


    def test(self,text):
        query_data_feature = self.loadData.features(text)

        model = joblib.load('model.pkl')
        # 预测新样本
        X_test = np.array(query_data_feature)
        y_pred = model.predict(X_test)
        y = self.idtolabel.get(y_pred[0])
        print(y_pred)  # 输出预测结果
        print(y)  # 输出预测结果







if __name__=="__main__":
    data = loadData()
    # test = data.features('MMD049一体化专核密码管理系统的主要操作流程是怎样的')
    # print(test)
    model = Bayes()
    # model.train_data()
    # model.train()
    model.test('TJM0006型端机保密机的配置选项有哪些？')




