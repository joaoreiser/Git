import tensorflow as tf
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt

num_epochs = 10
batch_size = 16

filename = "C:/Users/JGUILHERME/Documents/UFSC/Dissertação/Arquivos_Testes_Python/Plant_Dataset.tfrecords"

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
    return image, label

def Dataset_Generator(filename):
	dataset = tf.data.TFRecordDataset(filename)
	dataset = dataset.map(decode)
	dataset = dataset.map(normalize)
	dataset = dataset.shuffle(256)
	dataset = dataset.repeat(num_epochs)
	dataset = dataset.batch(batch_size)
	iterator = dataset.make_one_shot_iterator()
	#image_batch, label_batch = iterator.get_next()
	return iterator.get_next()


with tf.Session() as sess:
    image_batch, label_batch = Dataset_Generator(filename)
    for i in range(10):
        image, label = sess.run([image_batch, label_batch])
        print(np.shape(image))
        #print(image[5])
        data = image[5]
        print(np.shape(data))
        print(np.max(data))
        plt.imshow(data)
        print(label[5])
        plt.show()
