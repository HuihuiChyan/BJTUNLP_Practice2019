#encoding=utf-8

import tensorflow as tf
import data_helper
import numpy as np


class TextCNN(object):
    def __init__(self, vocab_size, seq_len, embedding_size,
                 num_classes, filter_sizes, num_filters):
        self.batch_size = 32
        self.inputs = tf.placeholder(tf.int32, [None, seq_len], name="inputs")
        self.labels = tf.placeholder(tf.float32, [None, num_classes], name="labels")
        self.dropout = tf.placeholder(tf.float32, name="dropout")
        """
        tf.placeholder(
            dtype,
            shape=None,
            name=None
        )
        dtype:数据类型，常用tf.float32,tf.float64
        shape:数据形状，默认None（一维），多维比如[None,3]表示行不定，列是3
        name:名称
        在网络构建graph的时候再模型中占位，分配必要的内存，运行模型时通过feed_dict()向占位符喂入数据。
        """


        # 1.构建中间层，单词转化成向量的形式
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # embedding表
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                 name="W")
            """
            tf.Variable(initializer, name)
            initializer:初始化参数
            
            tf.random_uniform(
                shape,
                minval = 0,
                maxval = None,
                dtype = tf.float32,
                seed = None,
                name = None
            )
            从均匀分布中输出随机值，范围在[minval,maxval)
            """

            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.inputs)
            """
            tf.nn.embedding_lookup(params, ids)
            选取一个张量里面索引对应的元素
            params:可以是张量也可以是数组等
            id:对应的索引
            """
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        #2.卷积层、激活层和池化层
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            """
            enumerate()将一个可遍历的数据对象组合为一个索引序列，同时列出数据和下标
            """
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                """
                tf.truncated_normal(
                    shape,
                    mean=0.0,
                    stddev=1.0,
                    dtype=tf.float32,
                    seed=None,
                    name=None
                )
                产生截断正态分布随机数，取值[mean-2*stddev, mean+2*stddev]
                shape:输出张量的维度
                mean:均值
                stddev:标准差
                """
                b = tf.Variable(tf.constant(0.1, shape=[num_filters], name="b"))

                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv"
                )
                """
                tf.nn.conv2d(
                    input,
                    filter,
                    strides,
                    padding,
                    use_cudnn_on_gpu=None,
                    data_format=None,
                    name=None
                )
                input:shape为[batch, in_height, in_weight,in_channel]
                filter:卷积核，shape为[filter_height, filter_weight, in_channel, out_channels]
                    in_channel要和input的in_channel一致，out_channel是卷积核数量
                strides:步长，[1, strides,strides, 1]第一位和最后一位固定是1
                padding:值为“SAME”考虑边界，不足的时候补0，“VALID”不考虑边界
                use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
                """
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                """
                tf.nn.relu(features, name=None)
                将大于0的保持不变，小于0的数置为0
                
                tf.nn.bias_add(value, bias, name=None)
                将一维的偏置项bias加到value上，向量与矩阵的每一行相加，得到结果和value大小相同
                """

                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, seq_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool"
                )
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        # 在最后一维concat
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # 3.进行dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout)
            """
            tf.nn.dropout(
                x,
                keep_prob,
                noise_shape=None,
                seed=None,
                name=None
            )
            在不同训练过程中随机扔掉一部分神经元
            keep_prob:float类型，每个元素被保留下来的概率，在初始化时是一个占位符，
                在run时设置具体的值
            noise_shape:一维的int32张量，代表随机产生“保留/丢弃”标志的shape
            """

        # 4.全连接层
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            """
            tf.argmax(input, axis)
            根据axis取值不同返回每行或每列最大值的索引
            axis=0：输出每列最大元素所在的索引数组
            axis=1：输出每行最大元素所在的索引数组
            """

        # 5.计算损失
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                             labels=self.labels)
            """
            tf.nn.softmax_cross_entropy_with_logits(
                logits,
                labels,
                name = None
            )
            logits：神经网络最后一层输出，大小[batch_size, num_classes]
            labels：样本实际的标签，大小同上
            第一步：对网络最后一层输出做softmax
            第二步：softmax的输出向量和样本实际标签做一个交叉熵
            """
            self.loss = tf.reduce_mean(losses)
            """
            tf.reduce_mean(
                input_tensor,
                axis = None,
                keep_dims = False,
                name = None
            )
            计算张量在某一维度上的平均值
            axis：指定的维度，如不指定，则计算所有元素的均值
            keep_dims：是否降维，True保持输入张量的形状，False降低维度
            """

        # 6.计算准确率
        # with tf.name_scope("accuracy"):
        #     correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels, 1))
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            

