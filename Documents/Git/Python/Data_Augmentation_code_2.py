
import cv2
import matplotlib.image as img
import matplotlib.pyplot as plt
import os
from random import randint

path_image = "C:/Users/JGUILHERME/Documents/UFSC/Dissertação/PlantVillage_Dataset/Tomato/"
path_dest = "C:/Users/JGUILHERME/Documents/UFSC/Dissertação/PlantVillage_Dataset/Tomato_Sick/"

folder = os.listdir(path_image)

names = ['', 'bact_', 'early_', 'late_', 'mold_', 'sept_', 'spider_', 'targ_', 'mosaic_', 'yellow_']

def data_augmentation(picture, sel):
    ab = 0
    if(sel == 0):
        ab = cv2.flip(picture, 0)
    elif(sel == 1):
        ab = cv2.flip(picture, 1)
    elif(sel == 2):
        ab = cv2.flip(picture, -1)
    elif(sel == 3):
        ab = cv2.blur(picture, (2, 2))
    elif(sel == 4):
        ab = cv2.medianBlur(picture, 5)
    elif(sel == 5):
        cropped = picture[10:240, 10:240]
        ab = cv2.resize(cropped, (256,256))
    else:
        cropped = picture[10:240, 10:240]
        ab = cv2.blur(cv2.flip(cropped, -1), (2,2))

    return ab

i = 0
for folders in folder:
    #print(folders)
    folders2 = os.listdir(path_image+folders)
    i = i+1
    for file in folders2:
        file_path = path_image + folders + '/' + file

        print(file_path)

        pict = img.imread(file_path)

        sel = randint(0, 6)

        ab = data_augmentation(pict, sel)

        p = file.find('.jpg')
        file = str(file[:p])

        dest = path_dest + names[i] + file

        name = dest + '_s.jpg'
        img.imsave(name,ab)
        