import tensorflow as tf
import os
import numpy as np
import matplotlib.image as img
import cv2

path_heal = "C:/Users/JGUILHERME/Documents/UFSC/Dissertação/PlantVillage_Dataset/Tomato_Healthy/"
path_sick = "C:/Users/JGUILHERME/Documents/UFSC/Dissertação/PlantVillage_Dataset/Tomato_Sick/"
path_record = "C:/Users/JGUILHERME/Documents/UFSC/Dissertação/PlantVillage_Dataset/Plant_Dataset_2.tfrecords"

heal = os.listdir(path_heal)
sick = os.listdir(path_sick)

def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

writer = tf.python_io.TFRecordWriter(path_record)

label = 0

for i in zip(heal,sick):
    img1 = path_heal+i[0]
    x = img.imread(img1,'jpg')
    x = cv2.cvtColor(x, cv2.COLOR_BGRA2BGR)
    x = x.tostring()
    y = 0

    data = {'image': wrap_bytes(tf.compat.as_bytes(x)),'label': wrap_int64(y)}
    # Wrap the data as TensorFlow Features.
    feature = tf.train.Features(feature=data)
    # Wrap again as a TensorFlow Example.
    example = tf.train.Example(features=feature)
    # Serialize the data.
    serialized = example.SerializeToString()
    # Write the serialized data to the TFRecords file.
    writer.write(serialized)

    img2 = path_sick + i[1]
    x = img.imread(img2,'jpg')
    x = cv2.cvtColor(x, cv2.COLOR_BGRA2BGR)
    x = x.tostring()
    y = 1

    data = {'image': wrap_bytes(x),'label': wrap_int64(y)}
    # Wrap the data as TensorFlow Features.
    feature = tf.train.Features(feature=data)
    # Wrap again as a TensorFlow Example.
    example = tf.train.Example(features=feature)
    # Serialize the data.
    serialized = example.SerializeToString()
    # Write the serialized data to the TFRecords file.
    writer.write(serialized)

    print(label)
    label+=1
writer.close()

