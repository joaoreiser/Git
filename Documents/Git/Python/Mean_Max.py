import numpy as np
import os

path = '/home/joaoreiser/Documentos/UFSC/Dissertação/Arquivos_Testes_Python/Power/power4/'

files = os.listdir(path)
files.sort()
for file in files:
    with open(path+file, 'r') as f:
        data = f.readlines()
    
    c = file.find('.')
    filename = file[:c]

    lista1 = []
    lista2 = []
    
    for d in data:
        #a = d.find('.')
        #num = int(d[:a])
        #print(d)
        num = int(d)
        if(num > 125):
            lista2.append(num)
        if(num > 90):
            lista1.append(num)
        else:
            continue

    print(filename)
    print(np.mean(lista1))
    print(np.mean(lista2))
    print(np.max(lista1))