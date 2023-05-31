#! /usr/bin/env python
# coding=utf-8


import os
import pickle
import tensorflow as tf
from nets import bilstm

load_mapping = True
mapping_file = "./config/mapping.pkl"
data_files = "./data/train.csv"
network_name = "BILSTM"
vocab_size = 10000
embedding_size = 100
num_units = 128
layers = 3
fc_units = [512, 1024, 512]
vector_size = 128
learning_rate = 0.0001
batch_size = 8
checkpoint_dir = "./running/model/bilstm"
model_file_name = "similarity.ckpt"
summary_dir = "./running/graph/bilstm"
train_summary_dir = os.path.join(summary_dir, "train")
dev_summary_dir = os.path.join(summary_dir, "dev")
max_num_checkpoints = 2
checkpoint_prefix = os.path.join(checkpoint_dir, model_file_name)

word_2_idx, idx_2_word, vocab_size = pickle.load(open(mapping_file, 'rb'))

pb_file = "./bilstm_similarity.pb"
ckpt_file = r".\running\model\bilstm\similarity.ckpt"
output_node_names = ["input/input_data",
                     "BILSTM/network/output/embedding/vector",
                     "BILSTM/loss/similarity"]

with tf.name_scope('input'):
    input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_data')

with tf.variable_scope("BILSTM"):
    # 2. 网络的前向过程构建
    with tf.variable_scope("network"):
        net = bilstm.Network(input_tensor=input, vocab_size=vocab_size,
                             embedding_size=embedding_size, num_units=num_units,
                             layers=layers, fc_units=fc_units, vector_size=vector_size)

    with tf.variable_scope("loss"):
        # a. 将文本转换得到的向量进行split拆分，还原文本信息
        # [N, vector_size], [N, vector_size]
        embedding1, embedding2 = tf.split(net.vector_embeddings, 2, 0)
        # b. 计算对应样本之间的相似度(只需要计算embedding1[i]和embedding2[i]的之间的相似度，其它不同行之间的相似度可以不考虑)
        # 最终得到N个相似度的值
        a = tf.reduce_sum(tf.multiply(embedding1, embedding2), axis=-1)  # [N,]
        b = tf.sqrt(tf.reduce_sum(tf.square(embedding1), axis=-1) + 1e-10)  # [N,]
        c = tf.sqrt(tf.reduce_sum(tf.square(embedding2), axis=-1) + 1e-10)  # [N,]
        similarity = tf.identity(a / tf.multiply(b, c), 'similarity')  # [N,], 取值范围: (-1,1)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                   input_graph_def=sess.graph.as_graph_def(),
                                                                   output_node_names=output_node_names)

with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())
