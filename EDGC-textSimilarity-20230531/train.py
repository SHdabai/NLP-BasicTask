# -- encoding:utf-8 --

from datetime import datetime
import os
import pickle
import tensorflow as tf

from nets import bilstm
from utils import data_helpers

tf.app.flags.DEFINE_boolean("is_train", True, "给定True或者False表示模型训练还是预测!!")

FLAGS = tf.app.flags.FLAGS


def train():
    load_mapping = True
    mapping_file = "./config/mapping.pkl"
    data_files = "./data/train.csv"
    network_name = "BILSTM"
    embedding_size = 100
    num_units = 128
    layers = 3
    fc_units = [512, 1024, 512]
    vector_size = 128
    learning_rate = 0.0001
    batch_size = 8
    num_epochs = 100
    checkpoint_every = 10
    checkpoint_dir = "./running/model/bilstm"
    model_file_name = "similarity.ckpt"
    summary_dir = "./running/graph/bilstm"
    train_summary_dir = os.path.join(summary_dir, "train")
    dev_summary_dir = os.path.join(summary_dir, "dev")
    max_num_checkpoints = 2
    checkpoint_prefix = os.path.join(checkpoint_dir, model_file_name)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(train_summary_dir):
        os.makedirs(train_summary_dir)
    if not os.path.exists(dev_summary_dir):
        os.makedirs(dev_summary_dir)

    # 一、数据加载(X1和X2中是原始的分词后的信息)
    X1, X2, Y = data_helpers.load_data_and_labels(data_files=data_files)

    # 2. 构建单词和id之间的映射关系
    if load_mapping and os.path.exists(mapping_file):
        tf.logging.info("从磁盘进行单词映射关系等数据的加载恢复!!!!")
        word_2_idx, idx_2_word, vocab_size = pickle.load(open(mapping_file, 'rb'))
    else:
        tf.logging.info("基于训练数据重新构建映射关系，并将映射关系保存到磁盘路径中!!!")
        # 加载
        word_2_idx, idx_2_word, vocab_size = data_helpers.create_mapping(datasets=X1 + X2)
        # 保存
        with open(mapping_file, 'wb') as writer:
            pickle.dump((word_2_idx, idx_2_word, vocab_size), writer)

    # 3. 文本数据转换为id来表示(这个过程中不进行填充，但是对于不在词表中的数据来讲，使用UNKNOWN进行替换)
    X1 = data_helpers.convert_text_2_idx(X1, word_2_idx)
    X2 = data_helpers.convert_text_2_idx(X2, word_2_idx)

    # 4. 批次数据的获取
    batches = data_helpers.batch_iter(
        data=list(zip(X1, X2, Y)),
        batch_size=batch_size,
        num_epochs=num_epochs
    )

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # 一、执行图的构建
            with tf.variable_scope(network_name):
                # 1. 占位符的构建
                with tf.variable_scope("placeholder"):
                    input_x1 = tf.placeholder(tf.int32, shape=[None, None])  # [N,T]
                    input_x2 = tf.placeholder(tf.int32, shape=[None, None])  # [N,T]
                    target = tf.placeholder(tf.int32, shape=[None])  # [N,]

                    # 将第一组和第二组合并
                    input = tf.concat([input_x1, input_x2], axis=0)  # [2N,T]

                # 2. 网络的前向过程构建
                with tf.variable_scope("network"):
                    net = bilstm.Network(input_tensor=input, vocab_size=vocab_size,
                                         embedding_size=embedding_size, num_units=num_units,
                                         layers=layers, fc_units=fc_units, vector_size=vector_size)

                # 3. 构建损失函数
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
                    # c. 如果target实际为1，那么希望similarity越大越好，如果为0，希望越小越好
                    logits = tf.concat([
                        tf.expand_dims(0 - similarity, -1),  # [N, 1] --> 表示的是不相似的可能性
                        tf.expand_dims(similarity, -1)  # [N,1] --> 表示的是相似的可能性
                    ], axis=-1)  # [N,2]
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits))
                    tf.losses.add_loss(loss)
                    loss = tf.losses.get_total_loss()
                    tf.summary.scalar('loss', loss)

                    predictions = tf.argmax(logits, axis=-1)
                    correct_predictions = tf.equal(predictions, tf.cast(target, predictions.dtype))
                    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
                    tf.summary.scalar('accuracy', accuracy)

                # 4. 构建优化器
                with tf.variable_scope("train_op"):
                    global_step = tf.train.get_or_create_global_step()
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                    train_op = optimizer.minimize(loss=loss, global_step=global_step)

                # 5、可视化对象构建
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, tf.get_default_graph())
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, tf.get_default_graph())
                summary_op = tf.summary.merge_all()

                # 5. 持久化相关操作的构建
                # 记录全局参数
                saver = tf.train.Saver(max_to_keep=max_num_checkpoints)

            # 二、模型的迭代训练
            # 1.模型初始化恢复
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Restore model weight from '{}'".format(checkpoint_dir))
                # restore：进行模型恢复操作
                saver.restore(sess, ckpt.model_checkpoint_path)
                # recover_last_checkpoints：模型保存的时候，我们会保存多个模型文件，默认情况下，模型恢复的时候，磁盘文件不会进行任何操作，为了保证磁盘中最多只有max_to_keep个模型文件，故需要使用下列API
                saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)

            # 2. 数据的遍历、然后进行训练
            def train_step(x1_batch, x2_batch, y_batch):
                feed_dict = {
                    input_x1: x1_batch,
                    input_x2: x2_batch,
                    target: y_batch
                }
                _, step, summaries, _loss, _accuracy = sess.run(
                    [train_op, global_step, summary_op, loss, accuracy],
                    feed_dict)
                time_str = datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, _loss, _accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x1_batch, x2_batch, y_batch, writer=None):
                feed_dict = {
                    input_x1: x1_batch,
                    input_x2: x2_batch,
                    target: y_batch
                }

                step, summaries, _loss, _accuracy = sess.run(
                    [global_step, summary_op, loss, accuracy],
                    feed_dict)
                time_str = datetime.now().isoformat()
                # step  步进  loss
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, _loss, _accuracy))
                if writer:
                    writer.add_summary(summaries, step)
                return _loss, _accuracy

            # d. 迭代所有批次
            for batch in batches:
                # 1. 将x和y分割开
                x1_batch, x2_batch, y_batch = batch
                # 2. 训练操作
                train_step(x1_batch, x2_batch, y_batch)
                # 3. 获取当前的更新的次数
                current_step = tf.train.global_step(sess, global_step)
                # 4. 进行模型持久化输出
                if current_step % checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            # e. 最终持久化
            path = saver.save(sess, checkpoint_prefix)
            print("Saved model checkpoint to {}\n".format(path))


def eval():
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

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(train_summary_dir):
        os.makedirs(train_summary_dir)
    if not os.path.exists(dev_summary_dir):
        os.makedirs(dev_summary_dir)

    # 一、数据加载(X1和X2中是原始的分词后的信息)
    X1, X2, Y = data_helpers.load_data_and_labels(data_files=data_files)

    # 2. 构建单词和id之间的映射关系
    if load_mapping and os.path.exists(mapping_file):
        tf.logging.info("从磁盘进行单词映射关系等数据的加载恢复!!!!")
        word_2_idx, idx_2_word, vocab_size = pickle.load(open(mapping_file, 'rb'))
    else:
        raise Exception("必须从磁盘恢复!!!")

    # 3. 文本数据转换为id来表示(这个过程中不进行填充，但是对于不在词表中的数据来讲，使用UNKNOWN进行替换)
    X1 = data_helpers.convert_text_2_idx(X1, word_2_idx)
    X2 = data_helpers.convert_text_2_idx(X2, word_2_idx)

    # 4. 批次数据的获取
    batches = data_helpers.batch_iter(
        data=list(zip(X1, X2, Y)),
        batch_size=batch_size,
        num_epochs=1
    )

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # 一、执行图的构建
            with tf.variable_scope(network_name):
                # 1. 占位符的构建
                with tf.variable_scope("placeholder"):
                    input_x1 = tf.placeholder(tf.int32, shape=[None, None])  # [N,T]
                    input_x2 = tf.placeholder(tf.int32, shape=[None, None])  # [N,T]
                    target = tf.placeholder(tf.int32, shape=[None])  # [N,]

                    # 将第一组和第二组合并
                    input = tf.concat([input_x1, input_x2], axis=0)  # [2N,T]

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
                    # 最终得到N个相似度的值（如果这里的相似度计算从夹角余弦相似度更改为距离相似度的话，那么搜索的时候就可以使用KDTree进行加速）
                    # 夹角余弦相似度
                    # a = tf.reduce_sum(tf.multiply(embedding1, embedding2), axis=-1)  # [N,]
                    # b = tf.sqrt(tf.reduce_sum(tf.square(embedding1), axis=-1) + 1e-10)  # [N,]
                    # c = tf.sqrt(tf.reduce_sum(tf.square(embedding2), axis=-1) + 1e-10)  # [N,]
                    # similarity = tf.identity(a / tf.multiply(b, c), 'similarity')  # [N,], 取值范围: (-1,1)

                    # 欧式距离转换的相似度
                    dist = tf.reduce_sum(tf.square(embedding1 - embedding2, 2), axis=-1)  # [N,]
                    similarity = tf.identity(1.0 / (dist + 1.0), 'similarity')  # [N,] 取值范围[0,1]

                    # c. 如果target实际为1，那么希望similarity越大越好，如果为0，希望越小越好
                    logits = tf.concat([
                        tf.expand_dims(0 - similarity, -1),  # [N, 1] --> 表示的是不相似的可能性
                        tf.expand_dims(similarity, -1)  # [N,1] --> 表示的是相似的可能性
                    ], axis=-1)  # [N,2]
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits))
                    tf.losses.add_loss(loss)
                    loss = tf.losses.get_total_loss()
                    tf.summary.scalar('loss', loss)

                    predictions = tf.argmax(logits, axis=-1)
                    correct_predictions = tf.equal(predictions, tf.cast(target, predictions.dtype))
                    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
                    tf.summary.scalar('accuracy', accuracy)

                # 3. 持久化相关操作的构建
                # 记录全局参数
                saver = tf.train.Saver(max_to_keep=max_num_checkpoints)

            # 二、模型的迭代训练
            # 1.模型初始化恢复
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Restore model weight from '{}'".format(checkpoint_dir))
                # restore：进行模型恢复操作
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("模型没有初始化好!!!")

            print(net.vector_embeddings)
            print(similarity)

            # 2. 数据的遍历、然后进行训练
            def dev_step(x1_batch, x2_batch, y_batch):
                feed_dict = {
                    input_x1: x1_batch,
                    input_x2: x2_batch,
                    target: y_batch
                }
                _logits, _loss, _accuracy = sess.run(
                    [logits, loss, accuracy],
                    feed_dict)
                time_str = datetime.now().isoformat()
                # step  步进  loss
                print("{}: loss {:g}, acc {:g}".format(time_str, _loss, _accuracy))
                return _loss, _accuracy

            _avg_loss = 0
            _avg_acc = 0
            count = 0
            # for batch in batches:
            #     x1_batch, x2_batch, y_batch = batch
            #     _loss, _acc = dev_step(x1_batch, x2_batch, y_batch)
            #     _avg_loss += _loss
            #     _avg_acc += _acc
            #     count += 1
            print("AVG LOSS:{:g}, AVG ACC:{:g}".format(_avg_loss / count, _avg_acc / count))


def main(_):
    if FLAGS.is_train:
        tf.logging.info("进行模型训练....")
        train()
    else:
        tf.logging.info("进行模型预测....")
        eval()


if __name__ == '__main__':
    tf.app.run()
