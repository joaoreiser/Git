import numpy as np
import os

path = '/home/joaoreiser/Documentos/UFSC/Dissertação/Arquivos_Testes_Python/Power/Files/'
path2 = '/home/joaoreiser/Documentos/UFSC/Dissertação/Arquivos_Testes_Python/Power/Files2/'


files = os.listdir(path)
for file in files:

    with open(path+file, 'r') as f:
        data = f.readlines()
    
    c = file.find('.')
    filename = file[:c]

    lista = []

    for d in data:
        a = d.find('.')
        lista.append(int(d[:a]))
    
    saver = np.array(lista)

    newfile = path2 + filename + '.csv'
    np.savetxt(newfile, saver, delimiter=',', fmt='% 4d')

