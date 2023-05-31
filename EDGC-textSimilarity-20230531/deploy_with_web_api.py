# -- encoding:utf-8 --

import pickle
from flask import Flask, request, jsonify
import tensorflow as tf
from utils import data_helpers


class SimilarityPredictor(object):

    def __init__(self):
        mapping_file = "./config/mapping.pkl"
        print("恢复映射关系.....")
        self.word_2_idx, self.idx_2_word, self.vocab_size = pickle.load(open(mapping_file, 'rb'))

        print("深度学习模型参数恢复.....")
        self.return_elements = [
            "input/input_data:0",
            "BILSTM/network/output/embedding/vector:0",
            "BILSTM/loss/similarity:0"
        ]
        self.pb_file = "./bilstm_similarity.pb"
        self.graph = tf.Graph()
        self.graph.as_default()
        self.return_tensors = self.read_pb_return_tensors()
        self.sess = tf.Session(graph=self.graph)

        print("恢复成功!!!!")

    def read_pb_return_tensors(self):
        with tf.gfile.FastGFile(self.pb_file, 'rb') as f:
            frozen_graph_def = tf.GraphDef()
            frozen_graph_def.ParseFromString(f.read())

        with self.graph.as_default():
            return_elements = tf.import_graph_def(frozen_graph_def,
                                                  return_elements=self.return_elements)
        return return_elements

    def convert_vector(self, text):
        """
        将文本转换为向量
        :param text:
        :return:
        """
        flag = False
        if isinstance(text, str):
            flag = True
            text = [text]

        vector = self.sess.run(self.return_tensors[1],
                               feed_dict={
                                   self.return_tensors[0]: data_helpers.pred_process_text(text, self.word_2_idx)
                               })
        if flag:
            return vector[0]
        else:
            return vector

    def calc_similarity(self, text_a, text_b):
        if not isinstance(text_a, str):
            raise Exception("参数必须为字符串!!!")
        if not isinstance(text_b, str):
            raise Exception("参数必须为字符串!!!")
        text = [text_a, text_b]
        simi = self.sess.run(self.return_tensors[2],
                             feed_dict={
                                 self.return_tensors[0]: data_helpers.pred_process_text(text, self.word_2_idx)
                             })
        return simi


if __name__ == '__main__':
    # =============================================
    # 下面为具体的模型部署代码
    # =============================================
    tf.logging.set_verbosity(tf.logging.DEBUG)
    predictor = SimilarityPredictor()

    # APP应用构建
    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False


    @app.route('/')
    @app.route('/index')
    def _index():
        return "你好，欢迎使用Flask Web API，进入文本相似度计算!!!"


    @app.route('/convert', methods=['POST'])
    def convert():
        tf.logging.info("基于给定的文本，将其转换为固定长度的向量.....")
        try:
            # 参数获取
            text = request.form.get("text")

            # 参数检查
            if text is None:
                return jsonify({
                    'code': 501,
                    'msg': '请给定参数text！！！'
                })

            # 直接调用预测的API
            vector = predictor.convert_vector(text)
            vector = list(map(float, vector))  # 数据类型转换
            return jsonify({
                'code': 200,
                'msg': '成功',
                'data': [
                    {
                        'text': text,
                        'vector': vector
                    }
                ]
            })
        except Exception as e:
            tf.logging.error("异常信息为:{}".format(e))
            return jsonify({
                'code': 502,
                'msg': '预测数据失败, 异常信息为:{}'.format(e)
            })


    @app.route('/similarity', methods=['POST'])
    def similarity():
        tf.logging.info("基于给定的文本，计算两个文本的相似度.....")
        try:
            # 参数获取
            text_a = request.form.get("text1")
            text_b = request.form.get("text2")

            # 参数检查
            if text_a is None:
                return jsonify({
                    'code': 501,
                    'msg': '请给定参数text1！！！'
                })
            if text_b is None:
                return jsonify({
                    'code': 501,
                    'msg': '请给定参数text2！！！'
                })

            # 直接调用预测的API
            simi = float(predictor.calc_similarity(text_a, text_b))
            return jsonify({
                'code': 200,
                'msg': '成功',
                'data': [
                    {
                        'text1': text_a,
                        'text2': text_b,
                        'similarity': simi
                    }
                ]
            })
        except Exception as e:
            tf.logging.error("异常信息为:{}".format(e))
            return jsonify({
                'code': 502,
                'msg': '预测数据失败, 异常信息为:{}'.format(e)
            })


    # 启动
    app.run(host='0.0.0.0', port=5000)
