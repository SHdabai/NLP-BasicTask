# -*- coding: utf-8 -*-

from mysql_client_tools import MysqlClient
from sql_config import mysql_config
import config
import jieba

import json
from flask import Flask,request
from flask_cors import CORS

app = Flask(__name__)
core = CORS(app)


# 访问地址 http://localhost:5000/?webData = ""&webProject=""
@app.route("/",methods=['GET'])
def matter_port():
    mc = MysqlClient(mysql_config)
    # from_web_data = request.args.to_dict() #从web出获得的所有数据
    # from_web_data = request.args.to_dict() #从web出获得的所有数据
    webData = request.args.get("webData",None)  #访问需要传入的参数
    webProject = request.args.get("webProject",None)  #访问需要传入的参数
    fromWebData = request.args.get("fromWebData","")  #访问需要传入的参数

    return_dict = {'return_code':200,'return_info':'请求成功'}
    # return json.dumps(return_dict,ensure_ascii=False)


    # todo 将数据库中的实体和相似词导入临时内存中
    def get_data():
        target_key = 'all_entity'
        if target_key in config.matter_dict:
            cache_data = config.matter_dict[target_key]

            return cache_data
        else:
            # mc = MysqlClient(mysql_config)
            # sql = 'SELECT * FROM emergency_management WHERE condition="密码忘记"'
            sql = 'SELECT * FROM emergency_management'

            try:
                result = mc.select_many(sql)

                p = []
                for row in result:
                    synonyms = row['synonyms']   #返回的是  哈哈|嘿嘿|呵呵|啊啊 形式数据。
                    synonyms_list = synonyms.split(sep = '|')  #返回的是同义词列表
                    p.extend(synonyms_list)
                    vocabulary = row['vocabulary']   #返回的是问题的关键词
                    p.append(vocabulary)
                config.matter_dict[target_key] = p

                return config.matter_dict[target_key]
            except:
                return "数据不全，请查证后进行查询"

    #todo 将关键词和同义词进行 hash 映射操作
    def getHash():

        # sql = 'SELECT * FROM emergency_management WHERE condition="密码忘记"'
        sql = 'SELECT * FROM emergency_management'

        try:
            result = mc.select_many(sql)

            vocabulary_dict = {}
            for row in result:
                # print(row)
                synonyms = row['synonyms']  # 返回的是  哈哈|嘿嘿|呵呵|啊啊 形式数据。
                synonyms_list = synonyms.split(sep='|')  # 返回的是同义词列表
                synonyms_list.append(row['vocabulary'])
                # print(synonyms_list)
                for i in synonyms_list:
                    if i not in vocabulary_dict:
                        vocabulary_dict[i] = row['vocabulary']
                    else:
                        continue
        except:
            return "数据库查询异常"

        return vocabulary_dict

    #TODO 判断两个列表是否有交集
    def inter(a,b):
        return list(set(a)&set(b))

    #TODO 进行分词匹配操作
    all_token = get_data() #分词加载动态词典
    all_dict = getHash() #数据库查询前进行同义词与词的映射

    for i in all_token:
        jieba.add_word(i)
    # jieba.add_word("人民共")
    f = jieba.lcut(webData + fromWebData)


    # 同义词与词的映射 后的键的列表
    m = [i for i in all_dict]
    n = inter(f,m) #判断输入描述与词典是否有交集

    if n :
        for j in f:

            if j not in all_dict:
                continue
            else:
                chat_vocabulary = all_dict[j]
                try:
                    sql = 'SELECT * FROM emergency_management WHERE vocabulary="{}"'.format(chat_vocabulary)
                    result = mc.select_one(sql)
                    # print(result)
                    if webProject == '"现场处置"':
                        site_disposal = result["site_disposal"]  #现场处置
                        return_dict["site_disposal"] = site_disposal
                        return_dict['return_code'] = 200
                        return_dict = json.dumps(return_dict, ensure_ascii=False)
                        return return_dict

                    elif webProject == '"应急处置"':
                        emergency = result["emergency"]  #应急处置
                        return_dict["emergency"] = emergency
                        return_dict['return_code'] = 200
                        return_dict = json.dumps(return_dict, ensure_ascii=False)
                        return return_dict

                    elif webProject == '"后续处置"':
                        follow_up = result["follow_up"]  #后续处置
                        return_dict["follow_up"] = follow_up
                        return_dict['return_code'] = 200
                        return_dict = json.dumps(return_dict, ensure_ascii=False)

                        return return_dict

                except:
                    error = "输入信息有错误，请查证后再次尝试"
                    return_dict["error"] = error
                    # return_dict = json.dumps("{}".format(error),ensure_ascii=False)
                    return_dict['return_code'] = 400
                    return_dict = json.dumps(return_dict,ensure_ascii=False)
                    return return_dict
                break

    else:
        error = "输入信息不存在，请查证后再次尝试"
        return_dict["error"] = error
        # return_dict = json.dumps("{}".format(error), ensure_ascii=False)
        return_dict['return_code'] = 400
        return_dict = json.dumps(return_dict, ensure_ascii=False)
        return return_dict

if __name__=="__main__":
    app.run(host='0.0.0.0',port=5000,debug=False)






