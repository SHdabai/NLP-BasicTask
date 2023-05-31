# -*- coding: utf-8 -*-

from mysql_client_tools import MysqlClient
from sql_config import mysql_config
import config
import jieba

#todo 将数据库中的实体和相似词导入临时内存中


mc = MysqlClient(mysql_config)
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
                follow_up = row['follow_up']   #返回的是  哈哈|嘿嘿|呵呵|啊啊 形式数据。
                follow_up_list = follow_up.split(sep = '|')  #返回的是同义词列表
                p.extend(follow_up_list)
                vocabulary = row['vocabulary']   #返回的是问题的关键词
                p.append(vocabulary)
            config.matter_dict[target_key] = p

            return config.matter_dict[target_key]
        except:
            print("数据不全，请查证后进行查询")

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
        return vocabulary_dict

    except:
        print("数据库查询异常")


#TODO 进行分词匹配操作
all_token = get_data() #分词加载动态词典
all_dict = getHash() #数据库查询前进行同义词与词的映射
pseg = "若密码卡丢失、失控或被敌获取，由军委办公厅机要局组织，视情对全网密钥管理系统自身相应算法参数进行更换"
# test = ["IC卡丢失","账户被盗"]
for i in all_token:
    jieba.add_word(i)
f = jieba.lcut(pseg)
print('要素分解：',f)
for j in f:
    if j not in all_dict:
        continue
    else:
        chat_vocabulary = all_dict[j]
        try:
            sql = 'SELECT * FROM emergency_management WHERE vocabulary="{}"'.format(chat_vocabulary)
            result = mc.select_one(sql)
            # print(result)

            site_disposal = result["site_disposal"]  #现场处置

            emergency = result["emergency"]  #应急处置

            follow_up = result["follow_up"]  #后续处置




        except:
            print("输入信息有错误，请查证后再次尝试")




if __name__=="__main__":
    # mc = MysqlClient(mysql_config)
    # # sql = 'SELECT * FROM emergency_management WHERE condition="密码忘记"'
    # sql = 'SELECT vocabulary FROM emergency_management WHERE condition="密码忘记"'
    # result = mc.select_one(sql)
    # print(result)
    test = get_data()
    print(test)









