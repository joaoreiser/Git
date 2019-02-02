import sys
#if necessary, insert the path for TF_Reader_2
#if not, delete next line
sys.path.insert(0, '/home/jupyter/joao.reiser/dissertacao/Accuracy/')

import tensorflow as tf
import time
import numpy as np
from TF_Reader_2 import Dataset_Generator_Train, Dataset_Generator_Test
import os
import pandas as pd

##parameters definitions## 

tf.set_random_seed(100)

SHAPE_HEIGHT = 256
SHAPE_WEIGHT = 256
NUMBER_OF_CHANNELS = 3

NUMBER_OF_EPOCHS = 20

FILTERS_LAYER_1 = 16

FULL_CONN = 2

BATCH_SIZE = 32

tf.reset_default_graph()
tf.set_random_seed(100)

##parameters definitions## 

##network definition##

def cnn(x):
    with tf.name_scope('conv1'):
        conv_layer_1 = tf.layers.conv2d(inputs=x, filters=FILTERS_LAYER_1, kernel_size=[3,3], strides=[2,2], padding='SAME', use_bias=True)
        batch_norm_1 = tf.layers.batch_normalization(inputs=conv_layer_1)
        activation_1 = tf.nn.relu(batch_norm_1)
        
    with tf.name_scope('pool1'):
        pooling_1 = tf.layers.max_pooling2d(inputs= activation_1, pool_size=[2,2], strides=2)

    with tf.name_scope('flatten'):
        flat_layer = tf.layers.flatten(pooling_1)

    with tf.name_scope('fully_connected'):
        y_conv = tf.layers.dense(inputs=flat_layer, units=FULL_CONN, activation=tf.nn.softmax, name='output', use_bias=True)

    return y_conv

x_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SHAPE_HEIGHT, SHAPE_WEIGHT, NUMBER_OF_CHANNELS], name='input')
y_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, FULL_CONN])

y_net = cnn(x_placeholder)

##network definition##

##training configurations##

with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_placeholder, logits=y_net))
with tf.name_scope('train'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(0.0005).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_net, 1), tf.argmax(y_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

##training configurations##

init = tf.global_variables_initializer()
saver = tf.train.Saver()

##server configurations##

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 136
config.inter_op_parallelism_threads = 1

##server configurations##

#training session

with tf.Session(config=config) as sess:
    sess.run(init)
    acc_mean = 0
    acc_mean_list = []
    times_list = []
    for i in range(NUMBER_OF_EPOCHS):
        start = time.time()

	#initialize the TFRecord files for training	
	
        image_batch_train, label_batch_train = Dataset_Generator_Train(BATCH_SIZE) 

	#training loop

        while(True): 		
            try: #try to read data from file
                image, label = sess.run([image_batch_train, label_batch_train])
                try:
                    sess.run(train_step, feed_dict={x: image, y: label})
                except:
                    t = time.time() - start
                    times_list.append(t)
                    continue                    
            except tf.errors.OutOfRangeError: #exception when all data is used
                acc_list = []

		#initialize the TFRecord files for testing	

                image_batch_test, label_batch_test = Dataset_Generator_Test(BATCH_SIZE)

		#test loop

                while(True): 
                    try: #try to read data from file
                        image_test, label_test = sess.run([image_batch_test, label_batch_test])
                        try:
                            acc_local_value = sess.run(accuracy, feed_dict={x: image_test, y: label_test})
                            acc_list.append(acc_local_value)
                        except:
                            continue
                    except tf.errors.OutOfRangeError: #exception when all data is used
                        break #force out of the test loop
                
                acc_mean = np.mean(acc_list)
                acc_mean_list.append(acc_mean)
                break #force out of the train loop

##training session##

##saving results##

total_time = np.mean(times_list)
ac = np.max(acc_mean_list)
name = os.path.basename(__file__)

columns = ['File', 'Time', 'Accuracy']

#data saved --> filename, mean time for epoch, max accuracy in test

data = [name, total_time, ac]

df = pd.DataFrame([data], columns=columns)
    
#insert the csv file for saving results
saving_file = '/home/jupyter/joao.reiser/dissertacao/Accuracy/my_csv.csv'

with open(saving_file, 'a') as f:
    df.to_csv(f, header=False)

##saving results##
