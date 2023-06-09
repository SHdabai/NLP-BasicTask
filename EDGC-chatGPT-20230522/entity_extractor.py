#!/usr/bin/env python3
# coding: utf-8
import os
import ahocorasick
from sklearn.externals import joblib
import jieba
import numpy as np


class EntityExtractor:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        data_dir  = './data/'
        # 路径
        self.vocab_path = os.path.join(cur_dir, 'data/vocab.txt')
        self.stopwords_path =os.path.join(cur_dir, 'data/stop_words.utf8')
        self.word2vec_path = os.path.join(cur_dir, 'data/merge_sgns_bigram_char300.txt')
        # self.same_words_path = os.path.join(cur_dir, 'DATA/同义词林.txt')
        self.stopwords = [w.strip() for w in open(self.stopwords_path, 'r', encoding='utf8') if w.strip()]

        # 意图分类模型文件
        self.tfidf_path = os.path.join(cur_dir, 'model/tfidf_model.pkl')
        self.nb_path = os.path.join(cur_dir, 'model/model.pkl')  #朴素贝叶斯模型


        self.tfidf_model = joblib.load(self.tfidf_path)
        self.nb_model = joblib.load(self.nb_path)


        #{"equipment":[], "use":[], "function":[], "composition":[],"fault_phenomenon":[],"configuration":[]}
        self.equipment_path = data_dir + 'equipment.txt'
        self.use_path = data_dir + 'use.txt'
        self.function_path = data_dir + 'function.txt'
        self.composition_path = data_dir + 'composition.txt'
        self.fault_phenomenon_path = data_dir + 'fault_phenomenon.txt'
        self.configuration_path = data_dir + 'configuration.txt'


        self.equipment_entities = [w.strip() for w in open(self.equipment_path, encoding='utf8') if w.strip()]
        self.use_entities = [w.strip() for w in open(self.use_path, encoding='utf8') if w.strip()]
        self.function_entities = [w.strip() for w in open(self.function_path, encoding='utf8') if w.strip()]
        self.composition_entities = [w.strip() for w in open(self.composition_path, encoding='utf8') if w.strip()]
        self.fault_phenomenon_entities = [w.strip() for w in open(self.fault_phenomenon_path, encoding='utf8') if w.strip()]
        self.configuration_entities = [w.strip() for w in open(self.configuration_path, encoding='utf8') if w.strip()]


        self.region_words = list(set(self.equipment_entities+self.use_entities+self.function_entities+self.composition_entities+
                                     self.fault_phenomenon_entities+self.configuration_entities))


        # 构造领域actree

        self.equipment_tree = self.build_actree(list(set(self.equipment_entities)))
        self.use_tree = self.build_actree(list(set(self.use_entities)))
        self.function_tree = self.build_actree(list(set(self.function_entities)))
        self.composition_tree = self.build_actree(list(set(self.composition_entities)))
        self.fault_phenomenon_tree = self.build_actree(list(set(self.fault_phenomenon_entities)))
        self.configuration_tree = self.build_actree(list(set(self.configuration_entities)))


        # {"equipment":[], "use":[], "function":[], "composition":[],"fault_phenomenon":[],"configuration":[]}

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

        self.idtolabel = {'0': 'query_solution', '1': 'query_fault_cause', '2': 'query_configuration',
                     '3': 'query_use', '4': 'query_composition', '5': 'query_parameter', '6': 'query_function',
                     '7': 'query_operation', }

    def build_actree(self, wordlist):
        """
        构造actree，加速过滤
        :param wordlist:
        :return:
        """
        actree = ahocorasick.Automaton()
        # 向树中添加单词
        for index, word in enumerate(wordlist):
            actree.add_word(word, (index, word))  #
        actree.make_automaton()
        return actree

    def entity_reg(self, question):
        """
        模式匹配, 得到匹配的词和类型。如疾病，疾病别名，并发症，症状
        :param question:str
        :return:

        """
        self.result = {}

        for i in self.equipment_tree.iter(question):

            '''
            i: print
            (word, (index, word))
            '''

            word = i[1][1]
            if "equipment" not in self.result:
                self.result["equipment"] = [word]
            else:
                self.result["equipment"].append(word)

        for i in self.use_tree.iter(question):
            word = i[1][1]
            if "use" not in self.result:
                self.result["use"] = [word]
            else:
                self.result["use"].append(word)

        for i in self.function_tree.iter(question):
            wd = i[1][1]
            if "function" not in self.result:
                self.result["function"] = [wd]
            else:
                self.result["function"].append(wd)


        for i in self.composition_tree.iter(question):
            wd = i[1][1]
            if "composition" not in self.result:
                self.result["composition"] = [wd]
            else:
                self.result["composition"] .append(wd)

        for i in self.fault_phenomenon_tree.iter(question):
            wd = i[1][1]
            if "fault_phenomenon" not in self.result:
                self.result["fault_phenomenon"] = [wd]
            else:
                self.result["fault_phenomenon"] .append(wd)

        for i in self.configuration_tree.iter(question):
            wd = i[1][1]
            if "configuration" not in self.result:
                self.result["configuration"] = [wd]
            else:
                self.result["configuration"].append(wd)


        return self.result
        # '''
        # return :
        # {"Disease":[], "Alias":[], "Symptom":[], "Complication":[]}
        # {"equipment":[], "use":[], "function":[], "composition":[],"fault_phenomenon":[],"configuration":[]}
        #
        # '''




    def find_sim_words(self, question):
        """
        当全匹配失败时，就采用相似度计算来找相似的词
        :param question:
        :return:
        """
        import re
        import string
        from gensim.models import KeyedVectors

        jieba.load_userdict(self.vocab_path)  #加载分词字典
        self.model = KeyedVectors.load_word2vec_format(self.word2vec_path, binary=False)

        sentence = re.sub("[{}]", re.escape(string.punctuation), question)
        sentence = re.sub("[，。‘’；：？、！【】]", " ", sentence)
        sentence = sentence.strip()


#TODO step-1:构建分词数据列表..................................................................
        words = [w.strip() for w in jieba.cut(sentence) if w.strip() not in self.stopwords and len(w.strip()) >= 2]



#TODO step-2:将分词列表中的数据与各个实体库进行相似度计算..........................................
        alist = [] #all 相似度
        for word in words:
            temp = [self.equipment_entities, self.use_entities, self.function_entities,
                    self.composition_entities,self.fault_phenomenon_entities,self.configuration_entities]
            #这里temp对应的是 这几个实体的加载列表 temp: [[], [], [], []]


            for i in range(len(temp)):
                flag = ''
                if i == 0:
                    flag = "equipment"
                elif i == 1:
                    flag = "use"
                elif i == 2:
                    flag = "function"
                elif i == 3:
                    flag = "composition"
                elif i == 4:
                    flag = "fault_phenomenon"
                else:
                    flag = "configuration"

                scores = self.simCal(word, temp[i], flag)  #将分词列表中的数据与各个实体库进行相似度计算

                alist.extend(scores)

        temp1 = sorted(alist, key=lambda k: k[1], reverse=True)

        if temp1:
            self.result[temp1[0][2]] = [temp1[0][0]]

    def editDistanceDP(self, s1, s2):
        """
        采用DP方法计算编辑距离
        :param s1:
        :param s2:
        :return:
        """
        m = len(s1)
        n = len(s2)
        solution = [[0 for j in range(n + 1)] for i in range(m + 1)]
        for i in range(len(s2) + 1):
            solution[0][i] = i
        for i in range(len(s1) + 1):
            solution[i][0] = i

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    solution[i][j] = solution[i - 1][j - 1]
                else:
                    solution[i][j] = 1 + min(solution[i][j - 1], min(solution[i - 1][j],
                                                                     solution[i - 1][j - 1]))
        return solution[m][n]

    def simCal(self, word, entities, flag):
        """
        计算词语和字典中的词的相似度

        相同字符的个数/min(|A|,|B|)   +  余弦相似度

        :param word: str
        :param entities:List
        :return:
        """
        a = len(word)
        scores = []
        for entity in entities:
            sim_num = 0 #问句切分实体与 对比实体的相似字符数

            b = len(entity)
            c = len(set(entity+word))

            temp = []

            for w in word:
                if w in entity:
                    sim_num += 1
            if sim_num != 0:
                score1 = sim_num / c  # overlap score 相似字符数目占两者的总的字符数比例
                temp.append(score1)
            try:
                score2 = self.model.similarity(word, entity)  # 余弦相似度分数
                temp.append(score2)
            except:
                pass
            score3 = 1 - self.editDistanceDP(word, entity) / (a + b)  # 编辑距离分数
            if score3:
                temp.append(score3)

            score = sum(temp) / len(temp)  #计算的是三者分数的平均值

            if score >= 0.7: #平均值大于0.7 即算为匹配到
                scores.append((entity, score, flag))

        scores.sort(key=lambda k: k[1], reverse=True)
        '''
		返回的是一个基于分数排序的列表信息，
        '''

        return scores

    def check_words(self, wds, sent):
        """
        基于特征词分类
        :param wds:
        :param sent:
        :return:
        """
        for wd in wds:
            if wd in sent:
                return True
        return False

    def tfidf_features(self, text, vectorizer):
        """
        提取问题的TF-IDF特征
        :param text:
        :param vectorizer:
        :return:
        """
        jieba.load_userdict(self.vocab_path)
        words = [w.strip() for w in jieba.cut(text) if w.strip() and w.strip() not in self.stopwords]
        sents = [' '.join(words)]

        tfidf = vectorizer.transform(sents).toarray()
        return tfidf

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

    def model_predict(self, x, model):
        """
        预测意图
        :param x:
        :param model:
        :return:
        """
        pred = model.predict(x)
        y = [self.idtolabel.get(pred[0])]


        return y

    #TODO 实体抽取主函数
    def extractor(self, question):

        #TODO step-1:对问句进行模式匹配：将问句中的切分实体，与对应领域实体匹配..............................
        # return:  {"equipment":[], "use":[], "function":[], "composition":[],"fault_phenomenon":[],"configuration":[]}
        self.entity_reg(question)

        '''
		return self.result
		
		这里对问句进行模式匹配：
		将问句中的切分实体，与对应领域实体匹配，返回的数据为
		{"Disease":[], "Alias":[], "Symptom":[], "Complication":[]}
		{"equipment":[], "use":[], "function":[], "composition":[],"fault_phenomenon":[],"configuration":[]}
		
		'''

        #TODO step-2:这里判断是否有匹配，没有即进行相似度匹配..................................
        if not self.result:
            self.find_sim_words(question)


        #TODO step-3: 意图预测...............................................................

        intentions = []  # 查询意图
        tfidf_feature = self.tfidf_features(question, self.tfidf_model)  #tfidf_进行特征转换

        other_feature = self.other_features(question) #这个是根据模板匹配的目的特征
        m = other_feature.shape #查看数据维度
        other_feature = np.reshape(other_feature, (1, m[0])) #数据特征广播

        feature = np.concatenate((tfidf_feature, other_feature), axis=1) #两者特征进行拼接


        predicted = self.model_predict(feature, self.nb_model) #模型进行预测 目的意图

        # print(predicted,'predicted---------------------------------')
        # ['query_period']predicted - --------------------------------
        intentions.append(predicted[0])




        types = []  # all实体类型
        for v in self.result.keys():
            types.append(v)
        # 设备名字 操作  操作内容  组成  组成内容   功能   功能内容    配置  配置内容  故障现象  故障原因  故障解决方案
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


        #TODO setp-4: 已知设备名字，查询操作 and 操作内容
        if self.check_words(self.use_qwds, question) and ('equipment' in types ):
            # 这里的check_words 主要是进行判断 self.symptom_qwds中的词是否在question中存在

            intention = "query_use"
            if intention not in intentions:
                intentions.append(intention)


        #TODO setp-5:  已知 故障现象，查询故障解决方案 and 设备名字
        if self.check_words(self.solution_qwds, question) and ('equipment' in types or 'fault_phenomenon'  in types):
            intention = "query_solution"
            if intention not in intentions:
                intentions.append(intention)

        #TODO setp-6:  已知 故障现象，查询故障原因  and 设备名字
        if self.check_words(self.fault_cause_qwds, question) and ('equipment' in types or 'fault_phenomenon' in types):
            intention = "query_fault_cause"
            if intention not in intentions:
                intentions.append(intention)


        #TODO setp-7:  已知设备名字，查询组成 and 组成内容
        if self.check_words(self.composition_qwds, question) and ('equipment' in types ):
        # if self.check_words(self.composition_qwds, question) and ('equipment' in types or 'Alias' in types):
            intention = "query_composition"
            if intention not in intentions:
                intentions.append(intention)
        #TODO setp-8:  已知设备名字，查询配置 and 配置内容
        if self.check_words(self.configuration_qwds, question) and ('equipment' in types ):
            intention = "query_configuration"
            if intention not in intentions:
                intentions.append(intention)

        #TODO setp-9:  已知设备名字，查询功能 and  功能内容
        if self.check_words(self.configuration_qwds, question) and ('function' in types ):
            intention = "query_function"
            if intention not in intentions:
                intentions.append(intention)

        self.result["intentions"] = intentions

        #TODO setp-final: 最终返回的数据应该是
        # {"equipment":[], "use":[], "function":[], "composition":[],"fault_phenomenon":[],"configuration":[],"intentions":[]}
        return self.result
		
		
		
		
