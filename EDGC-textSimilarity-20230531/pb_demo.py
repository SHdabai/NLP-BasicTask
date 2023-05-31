#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-01-20 16:06:06
#   Description :
#
# ================================================================
from time import time

import jieba
import pickle
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from utils import data_helpers


def read_pb_return_tensors(graph, pb_file, return_elements):
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
    return return_elements


return_elements = [
    "input/input_data:0",
    "BILSTM/network/output/embedding/vector:0",
    "BILSTM/loss/similarity:0"
]
pb_file = "./bilstm_similarity.pb"
mapping_file = "./config/mapping.pkl"
word_2_idx, idx_2_word, vocab_size = pickle.load(open(mapping_file, 'rb'))
text = "﻿怎么更改花呗手机号码"
text = jieba.lcut(text)
text = data_helpers.convert_text_2_idx([text], word_2_idx)  # [1,N]
graph = tf.Graph()

return_tensors = read_pb_return_tensors(graph, pb_file, return_elements)

start_time = time()
with tf.Session(graph=graph) as sess:
    vector = sess.run(return_tensors[1],
                      feed_dict={return_tensors[0]: text})
    print("=" * 100)
    print(np.shape(vector))
    print(vector)


def pred_process_text(sentences, word_2_id):
    if not isinstance(sentences, list):
        sentences = [sentences]

    # 1. 分词
    sentences = [jieba.lcut(sentence) for sentence in sentences]

    # 2. 单词id转换
    sentences = data_helpers.convert_text_2_idx(sentences, word_2_idx)

    # 3. 做填充
    max_length = max([len(sentence) for sentence in sentences])
    sentences = data_helpers.padding_value(sentences, max_length)

    return sentences


text1 = "﻿怎么更改花呗手机号码"
text2 = "我的花呗是以前的手机号码，怎么更改成现在的支付宝的号码手机号"
text3 = "如何得知关闭借呗"
text4 = "我的条件可以开通花呗借款吗"
# 比较text1和text2的相似度，比较text3和text4的相似度，而且计算相似度的时候样本序列数目必须是偶数
text = [
    text1, text3,
    text2, text4
]
text = pred_process_text(text, word_2_idx)
with tf.Session(graph=graph) as sess:
    simi = sess.run(return_tensors[2],
                    feed_dict={return_tensors[0]: text})
    print("=" * 100)
    print(simi)
