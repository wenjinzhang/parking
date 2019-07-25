import cv2
import os
import numpy as np

'''
checkSpaces will produce segment folder with pictures of cars
rawimage should be image string
'''
def checkSpaces(rawimage):
    image = cv2.imread(rawimage)
    # fileIn = open("parkingSpaces.txt","r")
    # data = fileIn.readlines()
    # keys_vert = data[0].strip().split(" ")
    # keys_horizontal = data[1].strip().split(" ")
    # keys_vert = [int(x) for x in keys_vert]
    # keys_horizontal = [int(x) for x in keys_horizontal]
    keys_vert = np.load("vertical.npy")
    keys_horizontal = np.load("horizontal.npy")
    
    buffer = 0
    # save segmented pictures
    if not os.path.exists("./segment"):
        os.mkdir("./segment")
    for y in range(len(keys_vert)-1):
        for x in range(0, len(keys_horizontal)-1, 2):
            cv2.imwrite("./segment/"+"{:04d}".format(x/2) + "_" + "{:04d}".format(y) + '.jpg', \
                image[keys_horizontal[x]-buffer: keys_horizontal[x+1]+buffer, keys_vert[y]-buffer:keys_vert[y+1]]+buffer)
