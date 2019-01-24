

import tensorflow as tf
import time
import numpy as np

import os
import pandas as pd

tf.set_random_seed(100)

shapeH = 256
shapeW = 256
channels = 3

filters1 = 32
filters2 = 32
filters3 = 32

full_conn = 2

batch = 1

input = np.ones([1,256,256,3])
output = [[1,0]]

tf.reset_default_graph()
tf.set_random_seed(100)

def cnn(x):
    with tf.name_scope('conv1'):
        conv1 = tf.layers.conv2d(inputs=x, filters=filters1, kernel_size=[3,3], strides=[2,2], padding='SAME', use_bias=True)
        batch1 = tf.layers.batch_normalization(inputs=conv1)
        actv1 = tf.nn.relu(batch1)
        
    with tf.name_scope('pool1'):
        pool1 = tf.layers.max_pooling2d(inputs= actv1, pool_size=[2,2], strides=2)
    
    with tf.name_scope('conv2'):
        conv2 = tf.layers.conv2d(inputs=pool1, filters=filters2, kernel_size=[3,3], strides=[2,2], padding='SAME', use_bias=True)
        batch2 = tf.layers.batch_normalization(inputs=conv2)
        actv2 = tf.nn.relu(batch2)
        
    with tf.name_scope('pool2'):
        pool2 = tf.layers.max_pooling2d(inputs= actv2, pool_size=[2,2], strides=2)
        
    with tf.name_scope('conv3'):
        conv3 = tf.layers.conv2d(inputs=pool2, filters=filters3, kernel_size=[3,3], strides=[2,2], padding='SAME', use_bias=True)
        batch3 = tf.layers.batch_normalization(inputs=conv3)
        actv3 = tf.nn.relu(batch3)
        
    with tf.name_scope('pool3'):
        pool3 = tf.layers.max_pooling2d(inputs= actv3, pool_size=[2,2], strides=2)

    with tf.name_scope('flatten'):
        h_flat = tf.layers.flatten(pool3)

    with tf.name_scope('fully_connected'):
        y_conv = tf.layers.dense(inputs=h_flat, units=full_conn, activation=tf.nn.softmax, name='output', use_bias=True)

    return y_conv

x = tf.placeholder(tf.float32, shape=[batch, shapeH, shapeW, channels], name='input')
y = tf.placeholder(tf.float32, shape=[batch, full_conn])

y_net = cnn(x)

with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_net))
with tf.name_scope('train'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(0.0005).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_net, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    sess.run(y_net, feed_dict={x: input, y: output})
    saver.save(sess, 'saves/BN_32_32_32_0_0')
