#!/usr/bin/env python3
# coding: utf-8
from py2neo import Graph



# import pycorrector
# from Translate.queryTranslate import translate
#
# # '胃痛该怎么治疗'
# # corrected_sent, detail = pycorrector.correct('少先队员因该为老人让坐')
# # print(corrected_sent, detail)
#
# # test = Dict(['胃痛该怎么治疗'])
# test = translate('胃痛该怎么治疗')
#
# print(test)

import ahocorasick


def make_AC(AC, word_set):
    for word in word_set:
        AC.add_word(word, word)
    return AC


def test_ahocorasick():
    '''
    ahocosick：自动机的意思
    可实现自动批量匹配字符串的作用，即可一次返回该条字符串中命中的所有关键词
    '''
    key_list = ["苹果", "香蕉", "梨", "橙子", "柚子", "火龙果", "柿子", "猕猴挑"]

    AC_KEY = ahocorasick.Automaton()
    AC_KEY = make_AC(AC_KEY, set(key_list))
    AC_KEY.make_automaton()

    print('AC_KEY:',AC_KEY)

    test_str_list = ["我最喜欢吃的水果有：苹果、梨和香蕉",
                     "我也喜欢吃香蕉，但是我不喜欢吃梨"]

    for content in test_str_list:
        name_list = set()
        for item in AC_KEY.iter(content):
            # 将AC_KEY中的每一项与content内容作对比，若匹配则返回
            # item : <class 'tuple'>: (11, '苹果')

            print('item:',item)
            name_list.add(item[1])

        name_list = list(name_list)
        print('name_list:',name_list)

        if len(name_list) > 0:
            print(content, "--->命中的关键词有：", "\t".join(name_list))



class AnswerSearching:
    def __init__(self):
        # self.graph = Graph("http://localhost:7474", username="neo4j", password="123456")
        # self.graph = Graph("http://192.168.1.9:7474", username="neo4j", password="123456")
        self.graph = Graph("http://192.168.1.8:7474", username="neo4j", password="neo4j")
        self.top_num = 10

    def question_parser(self, data):
        """
        主要是根据不同的实体和意图构造cypher查询语句
        :param data: {"Disease":[], "Alias":[], "Symptom":[], "Complication":[],'intentions':[]}
        data: {"equipment":[], "use":[], "function":[], "composition":[],"fault_phenomenon":[],"configuration":[],"intentions":[]}
        :return:
        """
        sqls = []
        if data:
            for intent in data["intentions"]:
                sql_ = {}
                sql_["intention"] = intent
                sql = []
                if data.get("equipment"):
                   sql = self.transfor_to_sql("equipment", data["equipment"], intent)
                elif data.get("use"):
                    sql = self.transfor_to_sql("use", data["use"], intent)
                elif data.get("function"):
                    sql = self.transfor_to_sql("function", data["function"], intent)
                elif data.get("composition"):
                    sql = self.transfor_to_sql("composition", data["composition"], intent)
                elif data.get("fault_phenomenon"):
                    sql = self.transfor_to_sql("fault_phenomenon", data["fault_phenomenon"], intent)

                elif data.get("configuration"):
                    sql = self.transfor_to_sql("configuration", data["configuration"], intent)

                if sql:
                    sql_['sql'] = sql
                    sqls.append(sql_)
        return sqls

    def transfor_to_sql(self, label, entities, intent):
        """
        将问题转变为cypher查询语句
        :param label:实体标签
        :param entities:实体列表
        :param intent:查询意图
        :return:cypher查询语句
        查询故障原因   query_fault_cause
        查询功能  query_function
        查询操作  query_use
        查询组成   query_composition
        查询解决方案  query_solution
        查询配置  query_configuration
        """
        if not entities:
            return []
        sql = []

        #TODO  step-1:查询操作...........................................................
        if intent == "query_use" and label == "equipment":
            sql = ["MATCH (d:equipment)-[:equipment_use]->(u:use)-[:use_UseDescription]->(s) " \
                   "WHERE d.name='{0}' RETURN d.name,s.name".format(e)
                   for e in entities]


if __name__ == "__main__":
    # test_ahocorasick()
    #
    # a = set("qwerty")
    # b = len("qwerty")
    # print(a)
    # print(b)
    e = 'JJP951E会议电视密码机'
    query = "MATCH (d:equipment)-[:equipment_FaultPhenomenon]->(f:fault_phenomenon)-[:faultphenomenon_content]->(s) " \
                   "WHERE d.name='{0}' RETURN d.name,s.name".format(e)

    searcher = AnswerSearching()
    graph = searcher.graph
    test = graph.run(query).data()
    print(test)




























