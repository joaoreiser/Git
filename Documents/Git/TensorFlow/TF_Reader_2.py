import tensorflow as tf
import numpy as np

#change the file path for your files

filename_train = "/home/joaoreiser/Documentos/UFSC/Dissertação/Arquivos_Testes_Python/Plant_Dataset_Train_80.tfrecords"
filename_test = "/home/joaoreiser/Documentos/UFSC/Dissertação/Arquivos_Testes_Python/Plant_Dataset_Test_20.tfrecords"

def decode(serialized_example):
    features = tf.parse_single_example(serialized_example, features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),})
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, shape=[256,256,3])
    label = tf.cast(features['label'], tf.int32)
    return image, label

def normalize(image, label):
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.one_hot(label,2)    
    return image, label

def Dataset_Generator_Train(batch_size):
    dataset_train = tf.data.TFRecordDataset(filename_train)
    dataset_train = dataset_train.map(decode)
    dataset_train = dataset_train.map(normalize)
    dataset_train = dataset_train.shuffle(256)
    dataset_train = dataset_train.batch(batch_size)
    iterator_train = dataset_train.make_one_shot_iterator()
    return iterator_train.get_next()

def Dataset_Generator_Test(batch_size):
    dataset_test = tf.data.TFRecordDataset(filename_test)
    dataset_test = dataset_test.map(decode)
    dataset_test = dataset_test.map(normalize)
    dataset_test = dataset_test.batch(batch_size)
    iterator_test = dataset_test.make_one_shot_iterator()
    return iterator_test.get_next()

