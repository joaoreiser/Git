import os
from mvnc import mvncapi as mvnc
import numpy as np
import cv2
import pandas as pd
import time

#open the image file, resize and reshape the picture
im = cv2.imread('')
im = cv2.resize(im, (256,256))
data = np.reshape(im,[1,256,256,3])

#look for the device (Movidius) connected
devices = mvnc.EnumerateDevices()

#open the file (neural network)
path_graph = ''
with open(path_graph, mode='rb') as f:
    graphfile = f.read()                        

#select the desired device, open connecting and allocate the graph on it
device = mvnc.Device(devices[0])
device.OpenDevice()
graph = device.AllocateGraph(graphfile)
            
print('Download to NCS')

#load the image on the device to perform an inference
graph.LoadTensor(data.astype(np.float16), 'inf')
#receive the result as an array of probabilities
output, userobj = graph.GetResult()
         
#deallocate the graph on the device and close the connecting
graph.DeallocateGraph()
device.CloseDevice()
