#!/usr/bin/env python3
# coding: utf-8
from py2neo import Graph


class AnswerSearching:
    def __init__(self):
        # self.graph = Graph("http://localhost:7474", username="neo4j", password="123456")
        self.graph = Graph("http://192.168.1.9:7474", username="neo4j", password="123456")
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

        #TODO  step-2:查询功能...........................................................
        if intent == "query_function" and label == "equipment":
            sql = ["MATCH (d:equipment)-[:equipment_function]->(f:function)-[:function_FunctionDescription]->(s) " \
                   "WHERE d.name='{0}' RETURN d.name,s.name".format(e)
                   for e in entities]


        #TODO  step-3:查询配置............................................................
        if intent == "query_configuration" and label == "equipment":
            sql = ["MATCH (d:equipment)-[:equipment_configuration]->(c:configuration)-[:configuration_ConfigurationDescription]->(s) " \
                   "WHERE d.name='{0}' RETURN d.name,s.name".format(e)
                   for e in entities]


        #TODO  step-4: 查询组成.............................................................
        if intent == "query_composition" and label == "equipment":
            sql = ["MATCH (d:equipment)-[:equipment_composition]->(c:composition)-[:composition_CompositionDescription]->(s) " \
                   "WHERE d.name='{0}' RETURN d.name,s.name".format(e)
                   for e in entities]



        #TODO  step-5: 查询故障原因
        if intent == "query_fault_cause" and label == "equipment":
            sql = ["MATCH (d:equipment)-[:equipment_FaultPhenomenon]->(f:fault_phenomenon)-[:faultphenomenon_FailureCause]->(s) " \
                   "WHERE d.name='{0}' RETURN d.name,s.name".format(e)
                   for e in entities]


        #TODO  step-6: 查询解决方案
        if intent == "query_solution" and label == "equipment":
            sql = ["MATCH (d:equipment)-[:equipment_FaultPhenomenon]->(f:fault_phenomenon)-[:faultphenomenon_content]->(s) " \
                   "WHERE d.name='{0}' RETURN d.name,s.name".format(e)
                   for e in entities]


        return sql

    def searching(self, sqls):
        """
        执行cypher查询，返回结果
        :param sqls:
        :return:str
        """
        final_answers = []
        for sql_ in sqls:
            intent = sql_['intention']
            queries = sql_['sql']
            answers = []
            for query in queries:
                ress = self.graph.run(query).data()
                answers += ress
            final_answer = self.answer_template(intent, answers)
            if final_answer:
                final_answers.append(final_answer)
        return final_answers




    def answer_template(self, intent, answers):
        """
        根据不同意图，返回不同模板的答案
        :param intent: 查询意图
        :param answers: 知识图谱查询结果
        :return: str
        查询故障原因   query_fault_cause
        查询功能  query_function
        查询操作  query_use
        查询组成   query_composition
        查询解决方案  query_solution
        查询配置  query_configuration

        """
        final_answer = ""
        if not answers:
            return ""
        #TODO 查询操作...........................................................
        if intent == "query_use":
            equipment_use_dic = {}
            for data in answers:
                d = data['d.name']
                s = data['s.name']
                if d not in equipment_use_dic:
                    equipment_use_dic[d] = [s]
                else:
                    equipment_use_dic[d].append(s)
            i = 0
            for k, v in equipment_use_dic.items():
                if i >= 10:
                    break
                final_answer += "疾病 {0} 的症状有：{1}\n".format(k, ','.join(sorted(list(set(v)))))
                i += 1


        #TODO 查询功能...........................................................
        if intent == "query_function":
            equipment_function_dic = {}
            for data in answers:
                d = data['d.name']
                s = data['s.name']
                if d not in equipment_function_dic:
                    equipment_function_dic[d] = [s]
                else:
                    equipment_function_dic[d].append(s)
            i = 0
            for k, v in equipment_function_dic.items():
                if i >= 10:
                    break
                final_answer += "疾病 {0} 的症状有：{1}\n".format(k, ','.join(sorted(list(set(v)))))
                i += 1

        #TODO 查询配置............................................................
        if intent == "query_configuration":
            equipment_configuration_dic = {}
            for data in answers:
                d = data['d.name']
                s = data['s.name']
                if d not in equipment_configuration_dic:
                    equipment_configuration_dic[d] = [s]
                else:
                    equipment_configuration_dic[d].append(s)
            i = 0
            for k, v in equipment_configuration_dic.items():
                if i >= 10:
                    break
                final_answer += "疾病 {0} 的症状有：{1}\n".format(k, ','.join(sorted(list(set(v)))))
                i += 1

        #TODO 查询组成.............................................................
        if intent == "query_composition":
            equipment_composition_dic = {}
            for data in answers:
                d = data['d.name']
                s = data['s.name']
                if d not in equipment_composition_dic:
                    equipment_composition_dic[d] = [s]
                else:
                    equipment_composition_dic[d].append(s)
            i = 0
            for k, v in equipment_composition_dic.items():
                if i >= 10:
                    break
                final_answer += "疾病 {0} 的症状有：{1}\n".format(k, ','.join(sorted(list(set(v)))))
                i += 1

        #TODO 查询故障原因.........................................................
        if intent == "query_fault_cause":
            faultphenomenon_content_dic = {}
            for data in answers:
                d = data['d.name']
                s = data['s.name']
                if d not in faultphenomenon_content_dic:
                    faultphenomenon_content_dic[d] = [s]
                else:
                    faultphenomenon_content_dic[d].append(s)
            i = 0
            for k, v in faultphenomenon_content_dic.items():
                if i >= 10:
                    break
                final_answer += "疾病 {0} 的症状有：{1}\n".format(k, ','.join(sorted(list(set(v)))))
                i += 1

        #TODO 查询解决方案...........................................................
        if intent == "query_solution":
            faultphenomenon_FailureCause_dic = {}
            for data in answers:
                d = data['d.name']
                s = data['s.name']
                if d not in faultphenomenon_FailureCause_dic:
                    faultphenomenon_FailureCause_dic[d] = [s]
                else:
                    faultphenomenon_FailureCause_dic[d].append(s)
            i = 0
            for k, v in faultphenomenon_FailureCause_dic.items():
                if i >= 10:
                    break
                final_answer += "疾病 {0} 的症状有：{1}\n".format(k, ','.join(sorted(list(set(v)))))
                i += 1



        return final_answer
