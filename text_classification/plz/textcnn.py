# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 21:42:45 2019

@author: dangrui
"""
import tensorflow as tf
import numpy as np

class TextCNN(object):
    
    def __init__(self,
                 sequence_length,
                 num_classes,
                 vocab_size,
                 embedding_size,
                 filter_sizes,
                 num_filters,
                 l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(tf.int32,[None,sequence_length],name='input_x')
        self.input_y = tf.placeholder(tf.int32,[None,2],name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')
        self.learning_rate = tf.placeholder(tf.float32,name='learning_rate')
        l2_loss = tf.constant(0.0)
        # embedding层
        with tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random.uniform([vocab_size,embedding_size],-1.0,1.0),
                                name='W',trainable=True)
            # [batch_size,sequence_length,embedding_size]
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # [batch_size,sequence_length,embedding_size,1]
            # 为了将其应用于conv2d，故需要维度类似于图片，即[batch_size,height,width,channels]
            # 最后的维度1就是channels
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            
            
        #conv2D  input: [batch, in_height, in_width, in_channels]
        pooled_outputs = []
        # 卷积和池化层(包含len(filter_sizes)个)
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s'%filter_size):
                # [filter_height,filter_width,filter,in_channels,out_channels]
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # [batch_size,sequence_length-filter_size+1,1,num_filters]
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # [batch_size,sequence_length-filter_size+1,1,num_filters]
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # [batch_size,1,1,num_filters]
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        # 合并所有pool的输出
        #print("filter_size:",len(filter_sizes))  3
        num_filters_total = num_filters * len(filter_sizes)
        # [batch_size,1,1,num_filter*len(filter_sizes)]
        self.h_pool = tf.concat(pooled_outputs, len(filter_sizes))
        # [bathc_size, num_filter*len(filter_sizes)]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        
        # Dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        # 输出分类
        
        with tf.name_scope('output'):
            self.dense1=tf.layers.dense(inputs=self.h_drop,units=2,activation=tf.nn.sigmoid)
        
        print("h_drop:",self.h_drop.shape)
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)# ??为啥对b做正则
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")#tf.argmax(,1)每一行最大索引
     
        # 计算loss
        with tf.name_scope("loss"):
            # loss
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            # 正则化后的loss
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float32"), name="accuracy")

