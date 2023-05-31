文本相似度
    NOTE:
        扩展的了解一下simhash、BM25、WMD
        https://blog.csdn.net/qq_33373858/article/details/90812282
        https://www.cnblogs.com/combfish/p/8126857.html
============================================================
    目标：
        目标1. 任意给定两个文本，能够返回着两个文本的相似度；
        目标2. 如果两个文本是相似的，那么转换后的向量也得相似，提供一个接口，可以对任何给定一个文本，转换为对应的向量。
    如何计算两个文本相似度？
        目标3. 任意给定一个文本，给我返回和这个文本相似度超过某个阈值的其它所有文本的相关信息。
        1. 将两个文本转换为向量
            词袋法
            TFIDF
            Embedding(NLP基础中所说的所有词向量转换方式都可以使用)
            Bert
            自己训练深度学习模型(两种损失的定义方式，第一种是分类损失，第二种三元组损失函数)
        2. 计算两个向量的相似度
            相似度的计算公式：
                欧式距离、夹角余弦距离、汉明距离
    深度学习模型如何构建？
        模型训练：
            输入：
                x1: 文档1，形状为[N,T]
                x2: 文档2, 形状为[N,T]
                y: 文档之间的相似度信息，形状为:[N,]
            模型结构：
                input_x1 = tf.placeholder(tf.float32, [None, None]) # shape:[N,T]
                input_x2 = tf.placeholder(tf.float32, [None, None]) # shape:[N,T]
                input = tf.concat([input_x1, input_x2], axis=0) # [2N,T]

                lengths = tf.reduce_sum(tf.sign(tf.abs(input)), -1) # [2N,]

                # 构建RNN Cell
                cell_fw = create_cell()
                cell_bw = create_cell()

                # 动态构建这个LSTM的结果
                (output_fw, output_bw), _ = tf.bi_dy_rnn(cell_fw, cell_bw, input, lengths, tf.float32)

                # 合并LSTM的输出
                output = tf.concat((output_fw, output_bw), axis=-1) # [2N,T,2U]

                # 做全连接转换，转换为固定维度大小的向量值
                output = tf.layers.dense(output, 1000) # [2N,T,1000]
                embeddings = tf.layers.dense(output, 256) # [2N,T,256]
                embeddings = tf.reduce_sum(embeddings, axis=1) # [2N, 256]
                embeddings = embeddings / tf.tile(tf.expand_dims(tf.cast(lengths, tf.float32), -1), [1, 128]) # [2N,128]
                embeddings = tf.layers.dense(embeddings, 128) # [2N,128]


                # 拆分tensor，分别得到文档1的向量和文档2的向量
                # [N,128], [N,128]
                embeddings1,embeddings2 = tf.split(embeddings, 2, 0)
                # 直接计算embeddings1和embeddings2的损失函数的值
                # 损失函数的构建
                    1. 可以当作分类的方式来构建
                    2. 使用三元组损失函数(要求有三个embeddings)
        模型预测
            输入：
                x: N个文本，形状为:[N,T]
            结构:
                input = tf.placeholder(tf.float32, [None, None]) # shape:[N,T]

                lengths = tf.reduce_sum(tf.sign(tf.abs(input)), -1) # [N,]

                # 构建RNN Cell
                cell_fw = create_cell()
                cell_bw = create_cell()

                # 动态构建这个LSTM的结果
                (output_fw, output_bw), _ = tf.bi_dy_rnn(cell_fw, cell_bw, input, lengths, tf.float32)

                # 合并LSTM的输出
                output = tf.concat((output_fw, output_bw), axis=-1) # [N,T,2U]

                # 做全连接转换，转换为固定维度大小的向量值
                output = tf.layers.dense(output, 1000) # [N,T,1000]
                embeddings = tf.layers.dense(output, 256) # [N,T,256]
                embeddings = tf.reduce_sum(embeddings, axis=1) # [N, 256]
                embeddings = tf.layers.dense(embeddings, 128) # [N,128]
                embeddings = embeddings / tf.tile(tf.expand_dims(tf.cast(lengths, tf.float32), -1), [1, 128]) # [N,128]


训练数据原始格式如下：(1表示相似，0表示不相似)
x1[0] x2[0] 1
x1[1] x2[1] 0
.....

============================================================
周六、周天思考一下：
    -1. Neo4j的安装、python使用
    -2. 文本的相似度计算剩下的代码的编写(模型训练部分的代码、最终模型部署是基于pb文件)
===================================================================================