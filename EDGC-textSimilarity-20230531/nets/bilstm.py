# -- encoding:utf-8 --

import tensorflow as tf

"""
       :param input: 
       """

"""
N: 批次样本数目
T: 序列长度
E：embedding操作后的词向量维度大小
U: LSTM转换之后输出的维度大小
"""


class Network(object):
    def __init__(self, input_tensor, vocab_size, embedding_size, num_units, layers, fc_units=None, vector_size=128):
        """
        :param input_tensor: 输入的Tensor对象，形状为:[N,T]
        :param vocab_size: 词汇表大小，是一个int的数字
        :param embedding_size: embedding转换后的单词向量维度大小
        :param num_units: LSTM中神经元的个数
        :param layers:  LSTM的层次
        :param fc_units:  对于LSTM输出值做FC全连接操作，全连接的神经元个数，可以是int或者list或者None
        :param vector_size:  最终转换得到的文本向量大小
        """
        # =====================================================
        # 下面的代码为设置属性信息
        # =====================================================
        self.input = input_tensor
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_units = num_units
        self.layers = layers
        self.fc_units = fc_units
        self.vector_size = vector_size

        # =====================================================
        # 下面的代码为网络构建
        # =====================================================
        tf.logging.info("开始构建网络.....")
        with tf.variable_scope("process"):
            self.lengths = tf.reduce_sum(tf.sign(tf.abs(self.input)), axis=-1, name='sequence_length')

        with tf.variable_scope("embedding"), tf.device("/cpu:0"):
            self.embedding = tf.get_variable(name='embedding_table', shape=[self.vocab_size, self.embedding_size])
            self.embedded_chars = tf.nn.embedding_lookup(self.embedding, self.input)  # [N,T] -> [N,T,E]

        with tf.variable_scope("rnn"):
            def cell(nu):
                return tf.nn.rnn_cell.BasicLSTMCell(nu)

            cell_fw = tf.nn.rnn_cell.MultiRNNCell(cells=[cell(self.num_units) for _ in range(self.layers)])
            cell_bw = tf.nn.rnn_cell.MultiRNNCell(cells=[cell(self.num_units) for _ in range(self.layers)])

            # b. 数据输入，并处理
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,  # 前向的RNN Cell
                cell_bw,  # 反向的RNN Cell
                inputs=self.embedded_chars,  # RNN的输入，形状为: [N, T, E]
                sequence_length=self.lengths,  # RNN输入的序列长度，形状为: [N,]
                dtype=tf.float32  # RNN初始化状态的时候会使用到的数据类型参数，默认的初始化状态为Zeros
            )

            # LSTM输出结果拼接
            rnn_output = tf.concat((output_fw, output_bw), axis=-1)  # ([N,T,U], [N,T,U]) -> [N,T,2U]

        with tf.variable_scope("output"):
            pre_output = rnn_output  # 赋值
            if self.fc_units is not None:
                if isinstance(self.fc_units, list):
                    for idx, units in enumerate(self.fc_units):
                        with tf.variable_scope("fc-{}".format(idx)):
                            pre_output = tf.layers.dense(pre_output, units=units, activation=tf.nn.relu)
                elif isinstance(self.fc_units, int):
                    with tf.variable_scope("fc-{}".format(0)):
                        pre_output = tf.layers.dense(pre_output, units=self.fc_units, activation=tf.nn.relu)

            with tf.variable_scope("avg_embedding"):
                units = pre_output.shape[-1]
                sum_embedding = tf.reduce_sum(pre_output, axis=1)  # [N, T, units] -> [N, units]
                avg_embedding = sum_embedding / tf.tile(
                    tf.expand_dims(tf.cast(self.lengths, tf.float32), -1),
                    [1, units]
                )  # [N, units]

            with tf.variable_scope("embedding"):
                result = tf.layers.dense(avg_embedding, self.vector_size)  # [N, vector_size]
                self.vector_embeddings = tf.identity(result, 'vector')  # 最终的文本向量


if __name__ == '__main__':
    input = tf.placeholder(dtype=tf.int32, shape=[None, None])
    network = Network(
        input_tensor=input,
        vocab_size=10000,
        embedding_size=100,
        num_units=64,
        layers=3,
        fc_units=[512, 1024, 512],
        vector_size=128
    )
    tf.summary.FileWriter('../running/graph', graph=tf.get_default_graph()).close()
