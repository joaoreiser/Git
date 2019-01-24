import os
from mvnc import mvncapi as mvnc
import numpy as np
import cv2
import pandas as pd
import time

path = '/home/joaoreiser/Documentos/UFSC/Dissertação/Arquivos_Testes_Python/Time_2/'
pastas1 = ['BN/', 'Convs/', 'Dense/']
pastas2 = ['4/','8/','16/','32/']
pasta3 = 'saves/'

im = cv2.imread('/home/joaoreiser/Documentos/UFSC/Dissertação/Arquivos_Testes_Python/Time_2/folha.jpg')
im = cv2.resize(im, (256,256))

data = np.reshape(im,[1,256,256,3])

devices = mvnc.EnumerateDevices()

for pasta1 in pastas1:
    for pasta2 in pastas2:
        pasta = path+pasta1+pasta2+pasta3
        files = os.listdir(pasta)
        files.sort()
        for file in files:
            print(file)
            fl = pasta+file
            with open(fl, mode='rb') as f:
                graphfile = f.read()                        
            
            device = mvnc.Device(devices[0])
            device.OpenDevice()
            graph = device.AllocateGraph(graphfile)
            
            print('Download to NCS')
            times = []
            
            for i in range(20):
                start = time.time()
                graph.LoadTensor(data.astype(np.float16), 'inf')
                output, userobj = graph.GetResult()
                stop = time.time()
                fulltime = stop-start
                times.append(fulltime)
            
            graph.DeallocateGraph()
            device.CloseDevice()
                        
            mean_time = np.mean(times)
            print(mean_time)
            
            columns = ['File', 'Time']
            data = [file, mean_time]
            df = pd.DataFrame([data], columns=columns)
            with open('/home/joaoreiser/Documentos/UFSC/Dissertação/Arquivos_Testes_Python/Time_2/Time_Results.csv', 'a') as f:
                df.to_csv(f, header=False)
