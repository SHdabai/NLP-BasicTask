# -- encoding:utf-8 --

import pymysql
import pickle
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from sklearn.neighbors import KDTree
from utils import data_helpers

tf.app.flags.DEFINE_boolean("--init",
                            False,
                            "是否做初始化操作，当设置为True的时候，进行初始化操作，就是将所有的问题转换为向量然后保存到数据库中")
tf.app.flags.DEFINE_string("--init_file",
                           "./data/sentences.txt",
                           "给定初始化的时候，对应的初始文件路径!!!")
FLAGS = tf.app.flags.FLAGS


def cosine_similarity(x, y):
    """
    计算两个向量x和y的相似度，要求x和y的维度大小是一致的
    :param x:
    :param y:
    :return:
    """
    # 1. 类型强制转换
    x = np.reshape(x, -1)
    y = np.reshape(y, -1)

    assert len(x) == len(y), "x和y大小不一致，不能计算相似度"
    # 2. 开始计算相似度
    a = np.sum(x * y)
    b = np.sqrt(np.sum(np.square(x))) * np.sqrt(np.sum(np.square(y)))
    return 1.0 * a / b


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

        print("开始进行数据库链接的相关获取操作....")
        self.conn = pymysql.connect(host="localhost", user="root", password="root",
                                    database="chat_robot", port=3306, charset="utf8")
        self.select_sql = "SELECT vectors,answer FROM tb_question q join tb_answer a ON a.id=q.answer_id"
        self.insert_sql = "INSERT INTO tb_question(question, vectors) VALUES(%s, %s)"
        self.reload_kdtree_algo = True
        self.algo = None
        self.answers = None
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

    def insert_question_vectors(self, questions, vectors):
        """
        原始的所有问题的信息转换为向量并保存
        :param questions:
        :param vectors:
        :return:
        """
        with self.conn.cursor() as cursor:
            for question, vector in zip(questions, vectors):
                cursor.execute(
                    self.insert_sql, (question, ','.join(map(str, vector)))
                )
            self.conn.commit()

    def search_answer_by_question(self, question, threshold):
        """
        基于给定的question文本以及阈值threshold选择返回对应的answer
        :param question:
        :param threshold:
        :return:
        """
        answer = None
        # 一、当前问题对应的128维的高阶向量
        vector = self.convert_vector(question)

        # 二、获取数据库中所有问题对应的高阶向量
        with self.conn.cursor() as cursor:
            cursor.execute(self.select_sql)
            vectors = []
            answers = []
            for tmp in cursor.fetchall():
                vectors.append(list(map(float, tmp[0].split(","))))
                answers.append(tmp[1])

        # 三、比较vector和vectors中的所有向量，根据相似度的计算，选择相似度最高的
        max_sim = -1
        max_idx = -1
        for idx, other_vector in enumerate(vectors):
            # 1. 计算相似度
            sim = cosine_similarity(vector, other_vector)
            # 2. 相似度比较
            if sim > max_sim:
                max_sim = sim
                max_idx = idx

        # 四、相似度的过滤
        if max_sim > threshold:
            # 最相似度的超过阈值，那么表示对应的answer就是我们需要的，否则返回None
            answer = answers[max_idx]
            tf.logging.info("问题:【{}】，最匹配的回复: 【{}】-【{}】，相关性为: 【{}】".format(
                question, max_idx, answer, max_sim))
        else:
            tf.logging.info("没有找到问题的最佳匹配，问题:【{}】，最大匹配相关性为: 【{}】-【{}】，阈值为: 【{}】".format(
                question, max_idx, max_sim, threshold))
        return max_sim, answer

    def fetch_kdtree_algo(self):
        if self.reload_kdtree_algo:
            with self.conn.cursor() as cursor:
                cursor.execute(self.select_sql)
                X = []
                answers = []
                for tmp in cursor.fetchall():
                    X.append(list(map(float, tmp[0].split(","))))
                    answers.append(tmp[1])
            self.algo = KDTree(X, leaf_size=10)
            self.answers = answers
            self.reload_kdtree_algo = False
        return self.algo, self.answers

    def search_answer_by_question_with_kdtree(self, question, threshold):
        """
        基于给定的question文本以及阈值threshold选择返回对应的answer
        NOTE: 只能针对欧式距离训练的模型
        :param question:
        :param threshold:
        :return:
        """
        answer = None
        # 一、当前问题对应的128维的高阶向量
        vector = self.convert_vector(question)  # [N,]

        # 二、获取模型以及标签值
        algo, answers = self.fetch_kdtree_algo()

        # 三、模型获取最解决的id以及距离
        dist, ind = algo.query([vector], k=1)
        dist = dist[0][0]
        ind = ind[0][0]

        # 四、距离转换为相似度
        max_sim = 1.0 / (dist + 1.0)

        # 四、相似度的过滤
        if max_sim > threshold:
            # 最相似度的超过阈值，那么表示对应的answer就是我们需要的，否则返回None
            answer = answers[ind]
            tf.logging.info("问题:【{}】，最匹配的回复: 【{}】-【{}】，相关性为: 【{}】".format(
                question, ind, answer, max_sim))
        else:
            tf.logging.info("没有找到问题的最佳匹配，问题:【{}】，最大匹配相关性为: 【{}】-【{}】，阈值为: 【{}】".format(
                question, ind, max_sim, threshold))
        return max_sim, answer


if __name__ == '__main__':
    # 一、构建对象
    tf.logging.set_verbosity(tf.logging.DEBUG)
    predictor = SimilarityPredictor()

    # 二、根据参数决定是否进行初始化操作
    if FLAGS.init:
        if tf.gfile.Exists(FLAGS.init_file):
            # 1. 加载所有数据
            X = []
            with open(FLAGS.init_file, 'r', encoding='utf-8-sig') as reader:
                tmp_x = []
                for line in reader:
                    line = line.strip()
                    tmp_x.append(line)
                    if len(tmp_x) >= 10:
                        X.append(tmp_x)
                        tmp_x = []
                if len(tmp_x) > 0:
                    X.append(tmp_x)

            # 2. 遍历所有的文本，然后获取对应的向量
            for texts in X:
                # a. 得到当前批次的文本对应的向量
                vectors = predictor.convert_vector(texts)
                # b. 将文本和向量转换后输出到数据库中
                # TODO: 正常情况下，数据库中是存在问答对的，也就是问题的字符串是存在的，所以这里实际上应该是更新操作，而不是插入。
                predictor.insert_question_vectors(texts, vectors)

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


    @app.route('/chat_robot', methods=['POST'])
    def chat_robot():
        tf.logging.info("基于给定的问题，使用检索的方式从数据库中获取匹配的回复.....")
        try:
            # 参数获取
            question = request.form.get("question")
            threshold = float(request.form.get("threshold", "0.9"))

            # 参数检查
            if question is None:
                return jsonify({
                    'code': 501,
                    'msg': '请给定参数question！！！'
                })

            # 直接调用预测的API
            max_similarity, answer = predictor.search_answer_by_question(question, threshold)
            if answer is None:
                return jsonify({
                    'code': 505,
                    'msg': '没有匹配的问题,最大相似度为:{}'.format(max_similarity)
                })
            else:
                return jsonify({
                    'code': 200,
                    'msg': '成功',
                    'data': [
                        {
                            'question': question,
                            'answer': answer,
                            'max_similarity': float(max_similarity)
                        }
                    ]
                })
        except Exception as e:
            tf.logging.error("异常信息为:{}".format(e))
            return jsonify({
                'code': 502,
                'msg': '预测数据失败, 异常信息为:{}'.format(e)
            })


    @app.route('/chat_robot2', methods=['POST'])
    def chat_robot2():
        tf.logging.info("基于给定的问题，使用检索的方式从数据库中获取匹配的回复（KDTree加速）.....")
        try:
            # 参数获取
            question = request.form.get("question")
            threshold = float(request.form.get("threshold", "0.9"))

            # 参数检查
            if question is None:
                return jsonify({
                    'code': 501,
                    'msg': '请给定参数question！！！'
                })

            # """
            # # 修改为完整的聊天机器人的流程：
            #     -1. 先做意图识别（predictor_model11）
            #     -2. 再根据意图类别，调用不同的模型进行预测，产生预测结果(回复结果)
            #         predictor_model21 --> 意图1/业务1/可以是规则匹配、检索、生成式模型
            #         predictor_model22 --> 意图2/业务2/可以是规则匹配、检索、生成式模型
            #         predictor_model23 --> 意图3/业务3/可以是规则匹配、检索、生成式模型
            #         predictor_model24 --> 闲聊模块/Seq2Seq生成式模型
            #     -3. 如果回复结果为None，那么调用生成式的模型产生回复结果。
            # """
            # type = predictor_model11.predict(question, threshold)
            # result = None
            # if type == '意图1':
            #     result = predictor_model21.predict(question, threshold)
            # elif type == '意图2':
            #     result = predictor_model22.predict(question, threshold)
            # elif type == '意图3':
            #     result = predictor_model23.predict(question, threshold)
            # if result is None:
            #     result = predictor_model24.predict(question, threshold)
            # # result处理返回即可

            # 直接调用预测的API
            max_similarity, answer = predictor.search_answer_by_question_with_kdtree(question, threshold)
            if answer is None:
                return jsonify({
                    'code': 505,
                    'msg': '没有匹配的问题,最大相似度为:{}'.format(max_similarity)
                })
            else:
                return jsonify({
                    'code': 200,
                    'msg': '成功',
                    'data': [
                        {
                            'question': question,
                            'answer': answer,
                            'max_similarity': float(max_similarity)
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
