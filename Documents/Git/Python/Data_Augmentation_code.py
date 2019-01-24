
import cv2
import matplotlib.image as img
import matplotlib.pyplot as plt
import os

path_image = "C:/Users/JGUILHERME/Documents/UFSC/Dissertação/PlantVillage_Dataset/gerar1/"
path_dest = "C:/Users/JGUILHERME/Documents/UFSC/Dissertação/PlantVillage_Dataset/gerar2/"

folder = os.listdir(path_image)

names = ['',  'health_', 'bact_', 'early_', 'late_', 'mold_', 'sept_', 'spider_', 'targ_', 'mosaic_', 'yellow_']

def data_augmentation(picture):
    flip1 = cv2.flip(picture, 0)
    flip2 = cv2.flip(picture, 1)
    flip3 = cv2.flip(picture, -1)
    blur1 = cv2.blur(picture, (2, 2))
    median = cv2.medianBlur(picture, 5)
    cropped = picture[10:240, 10:240]
    cropped = cv2.resize(cropped, (256,256))
    mix = cv2.blur(cv2.flip(cropped, -1), (2,2))

    return flip1, flip2, flip3, blur1, median, cropped, mix

i = 0
for folders in folder:
    #print(folders)
    folders2 = os.listdir(path_image+folders)
    i = i+1
    for file in folders2:
        file_path = path_image + folders + '/' + file

        print(file_path)

        pict = img.imread(file_path)

        flip1, flip2, flip3, blur1, median, cropped, mix = data_augmentation(pict)

        p = file.find('.jpg')
        file = str(file[:p])

        dest = path_dest + names[i] + file

        name = dest + '_f1.jpg'
        img.imsave(name,flip1)
        name = dest + '_f2.jpg'
        img.imsave(name, flip2)
        name = dest + '_f3.jpg'
        img.imsave(name, flip3)
        name = dest + '_b1.jpg'
        img.imsave(name, blur1)
        name = dest + '_m.jpg'
        img.imsave(name, median)
        name = dest + '_crop.jpg'
        img.imsave(name, cropped)
        name = dest + '_mix.jpg'
        img.imsave(name, mix)