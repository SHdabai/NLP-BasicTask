#!/usr/bin/env python3
# coding: utf-8

from entity_extractor import EntityExtractor
from search_answer import AnswerSearching
from flask import Flask,request,jsonify
from flask_cors import CORS

import pycorrector
from Translate.queryTranslate import translate


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
core = CORS(app)


class KBQA:
    def __init__(self):
        self.extractor = EntityExtractor()
        self.searcher = AnswerSearching()

    def qa_main(self, input_str):
        answer = "Sorry, I don't know your problem, I will try to improve in the future."
        entities = self.extractor.extractor(input_str)  # 抓取实体
        if not entities:
            return answer

        sqls = self.searcher.question_parser(entities)  # 根据实体和意图构造 图谱查询语句

        final_answer = self.searcher.searching(sqls)  # 查询 返回结果

        if not final_answer:
            return answer
        else:
            return '\n'.join(final_answer)


@app.route('/')
@app.route('/index')
def _index():
    return "welcome chat"

@app.route('/chat',methods=["GET","POST"])

def chat():

    try:
        query = request.form.get('query')
        class_id = request.form.get('id')
        if query is None:
            return jsonify({
                'code':501,
                'msg':'please input query'
            })
        if class_id is None:
            return jsonify({
                'code': 501,
                'msg': 'please input class_id'
            })


        if class_id == 1:
            Query = translate(query)
        else:
            Query, detail = pycorrector.correct(query)

        handler = KBQA()

        answer = handler.qa_main(Query)
        return jsonify({
            'code':200,
            'msg':'success',
            'data':[{
                'Query': Query,
                'language':class_id,
                'answer':answer
            }]
        })
    except Exception as e:
        return jsonify({
            'code':502,
            'msg':f'The input data is abnormal, and the abnormal information is{e}'
        })




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=50008)
