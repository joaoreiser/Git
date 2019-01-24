import os
from mvnc import mvncapi as mvnc
import numpy as np
import cv2
import pandas as pd
import time

path = '/home/joaoreiser/Documentos/UFSC/Dissertação/Arquivos_Testes_Python/Time_2/'

im = cv2.imread('/home/joaoreiser/Documentos/UFSC/Dissertação/Arquivos_Testes_Python/Time_2/folha.jpg')
im = cv2.resize(im, (256,256))

data = np.reshape(im,[1,256,256,3])

devices = mvnc.EnumerateDevices()

#pasta1 = 'BN/'
#pasta1 = 'Convs/'
pasta1 = 'Dense/'
#pasta2 = '4/'
#pasta2 = '8/'
pasta2 = '16/'
#pasta2 = '32/'
file = 'DENSE_128_128_128_16_16'

#fl = path + pasta1 + pasta2 + 'saves/' + file + '.graph'
fl = path + 'saves/' + file + '.graph'

with open(fl, mode='rb') as f:
    graphfile = f.read()                        
            
device = mvnc.Device(devices[0])
device.OpenDevice()
graph = device.AllocateGraph(graphfile)
            
print('Download to NCS')
print(file)
times = []

val = []

for i in range(10):
    graph.LoadTensor(data.astype(np.float16), 'inf')
    output, userobj = graph.GetResult()
    #valores = graph.GetGraphOption(mvnc.GraphOption.TIME_TAKEN)
    #val.append(valores)
'''            
graph.DeallocateGraph()
device.CloseDevice()

leng = np.shape(val)[1]

valores_final = []
for e in range(leng):
    f = 0
    for i in range(20):
        f += val[i][e]
    valores_final.append(f/20)    

print(np.sum(valores_final))

columns = ['File', 'Time']
data = [file, valores_final]
df = pd.DataFrame([data], columns=columns)
with open('/home/joaoreiser/Documentos/UFSC/Dissertação/Arquivos_Testes_Python/Time_2/Time_All_2.csv', 'a') as f:
    df.to_csv(f, header=False)
'''