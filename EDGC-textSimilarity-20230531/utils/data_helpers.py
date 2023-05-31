import os
import numpy as np
import pandas as pd
import re
import jieba
import itertools
from collections import Counter

PAD = "<PAD>"
UNKNOWN = "<UNKNOWN>"

# 添加自定义词典
JIEBA_WORD_DICT_PATH = "./config/jieba_word_split.dict"
if os.path.exists(JIEBA_WORD_DICT_PATH):
    jieba.load_userdict(JIEBA_WORD_DICT_PATH)


# 清洗字符串，字符切分
def clean_str(string):
    if not isinstance(string, str):
        if np.isnan(string):
            string = "unknown"
        else:
            string = str(string)

    string = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9(),.!?，。？！、“”\'\`]", " ", string)  # 考虑到中文
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# 分词的函数
def split_fn(string):
    return jieba.lcut(string)


# 给数据添加标签
def load_data_and_labels(data_files):
    """
    加载csv给定文件的数据，列的顺序为idx、text1、text2、label
    label取值1表示text1和text2文本相似，取值为0表示不相似。
    :param data_files:
    :return:
    """
    if isinstance(data_files, str):
        data_files = [data_files]

    X1 = []
    X2 = []
    Y = []

    for data_file in data_files:
        # 1. 数据加载
        df = pd.read_csv(data_file, sep='\t', header=None, names=['idx', 'text1', 'text2', 'label'])

        # 2. 数据遍历
        for _, text1, text2, label in df.values:
            # 字符串清洗的操作
            text1 = clean_str(text1)
            text2 = clean_str(text2)

            # 添加到集合中(添加的是分词之后的结果)
            X1.append(split_fn(text1))
            X2.append(split_fn(text2))
            Y.append(label)

    # 做一个numpy的转换
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    Y = np.asarray(Y)
    return X1, X2, Y


# 构建单词和id之间的映射关系
def create_mapping(datasets):
    """
    基于给定的集合进行单词id之间的映射关系的获取
    :param datasets:  是一个集合，集合内部是字符串
    :return:
    """
    # 1. 统计各个单词出现的次数
    vocab_size = 0
    total_word_count = 0
    word_2_count = {}
    for words in datasets:
        for word in words:
            if word not in word_2_count:
                vocab_size += 1  # 新单词就累加1
                word_2_count[word] = 0
            word_2_count[word] += 1  # 每出现一次，累加1
            total_word_count += 1

    # 2. 添加两个特殊值: PAD、UNKNOWN
    word_2_count[UNKNOWN] = total_word_count + 1  # 1
    word_2_count[PAD] = total_word_count + 2  # 0
    vocab_size += 2  # 两个特殊的单词

    # 3. 按照出现的次数，降序排列，出现次数越多的单词，就放到前面，那么转换id就越小
    word_2_idx = {}
    idx_2_word = {}
    idx = 0
    for word, count in sorted(word_2_count.items(), key=lambda t: -t[1]):
        word_2_idx[word] = idx
        idx_2_word[idx] = word
        idx += 1

    return word_2_idx, idx_2_word, vocab_size


# 将文本单词数据转换为id的形式来进行展示
def convert_text_2_idx(X, word_2_idx):
    """
    对集合X中的每个序列进行转换，X是一个集合，集合中是样本序列，每个样本序列又是一个集合，内部是一个一个的单词
    word_2_idx是单词和id之间的映射关系
    :param X:
    :param word_2_idx:
    :return:
    """
    # 1. 构建变量
    new_X = []
    unknown_idx = word_2_idx[UNKNOWN]

    # 2. 遍历
    for x in X:
        new_x = []
        for word in x:
            # 如果单词word有对应的id，获取对应id，如果没有获取默认值为unknown_idx
            idx = word_2_idx.get(word, unknown_idx)
            # idx添加到临时的集合中
            new_x.append(idx)
        # 将一个文本对应的所有单词id添加到最终的集合X
        new_X.append(new_x)

    # 转换为numpy
    new_X = np.asarray(new_X)
    return new_X


# 进行数据填充
def padding_value(X, max_length):
    result = []
    for x in X:
        new_x = np.zeros(shape=max_length, dtype=np.int32)  # 使用0填充数据
        for idx, value in enumerate(x):
            new_x[idx] = value
        result.append(list(new_x))
    return np.asarray(result, dtype=np.int32)


def pred_process_text(sentences, word_2_idx):
    if not isinstance(sentences, list):
        sentences = [sentences]

    # 1. 分词
    sentences = [jieba.lcut(sentence) for sentence in sentences]

    # 2. 单词id转换
    sentences = convert_text_2_idx(sentences, word_2_idx)

    # 3. 做填充
    max_length = max([len(sentence) for sentence in sentences])
    sentences = padding_value(sentences, max_length)

    return sentences


# 取出样本数据
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    # 一个epoch里面有多少个bachsize
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            # 传给permutation一个矩阵，它会返回一个洗牌后的矩阵副本
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            batch_x1, batch_x2, batch_y = zip(*shuffled_data[start_index:end_index])
            # 将batch_x1和batch_x2中的所有文本的长度设置为相同长度，也就是进行填充
            # 1. 获取序列最长长度为多少
            max_length = max(
                max([len(sentence) for sentence in batch_x1]),
                max([len(sentence) for sentence in batch_x2])
            )

            # 2. 遍历数据进行填充
            new_batch_x1 = padding_value(batch_x1, max_length)
            new_batch_x2 = padding_value(batch_x2, max_length)

            yield new_batch_x1, new_batch_x2, batch_y
