import cv2
import os
import numpy as np
import shutil


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
    keys_vert = np.load("../setupData/vertical.npy")
    keys_horizontal = np.load("../setupData/horizontal.npy")
    print("\n\n")
    print("checkspaces")
    print("keys_vertical: ")
    print(keys_vert)
    print("keys_horizontal: ")
    print(keys_horizontal)
    

    buffer = 0
    # save segmented pictures
    if not os.path.exists("../segment"):
        os.mkdir("../segment")

    # dirPath = '../segmentData'
    # # Delete all contents of a directory using shutil.rmtree() and handle exceptions
    # try:
    #     shutil.rmtree(dirPath)
    # except:
    #     print('Error while deleting directory')

    for y in range(len(keys_vert)-1):
        for x in range(0, len(keys_horizontal)-1, 2):
            cv2.imwrite("../segment/{:04d}_{:04d}".format(int(x/2), int(y)) + '.jpg', \
                image[keys_horizontal[x]-buffer: keys_horizontal[x+1]+buffer, keys_vert[y]-buffer:keys_vert[y+1]]+buffer)
