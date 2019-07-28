import numpy as np
import argparse
import cv2
import operator
import os
from transform_helper import grouper
from transform_helper import partition
from transform_helper import quickSort
from transform_helper import rgb_mask
 
'''
split creates horizontal.npy and vertical.npy which are the vertical and horizontal arrays of lines
that define the parking lot
imagestr should be image string
row is the number of rows in the image
column is the number of columns in the image
'''
def split(imagestr, row, column):
    image = cv2.imread(imagestr)
    mask = rgb_mask(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    gaus = cv2.GaussianBlur(gray_mask, (9, 9), 0)
    canny = cv2.Canny(gaus,50,100)
    hough = cv2.HoughLinesP(canny, rho=1, theta=np.pi / 180, threshold=15, minLineLength=15, maxLineGap=12)
    houghPic = np.copy(mask)
    # draw hough lines onto picture
    for line in hough:
        x1, y1, x2, y2 = line[0]
        cv2.line(houghPic, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # filter vertical and horizontal lines
    vertical = []
    horizontal = []
    for line in hough:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            vertical.append(line)
        else:
            m = (y2 - y1) / (x2 - x1)
            if m > 10 or m < -10:
                vertical.append(line)
            elif m<0.1 and m>-0.1:
                horizontal.append(line)

    #draw vertical and horizontal lines on picture
    vertPic = np.copy(mask)
    for line in vertical:
        x1, y1, x2, y2 = line[0]
        cv2.line(vertPic, (x1, y1), (x2, y2), (0, 255, 0), 3)

    horizPic = np.copy(mask)
    for line in horizontal:
        x1, y1, x2, y2 = line[0]
        cv2.line(horizPic, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # sorts vertical in order of increasing x cord and 
    # horizontal in order of increasing y cord
    quickSort(vertical, 0, len(vertical)-1, 0)
    quickSort(horizontal, 0, len(horizontal)-1, 1)

    # sort endpoints of vertical lines which will be used to find 
    # more horizontal lines 
    vertical_endpoints = []
    for line in vertical:
        x1, y1, x2, y2 = line[0]
        vertical_endpoints.append(y1)
        vertical_endpoints.append(y2)
    vertical_endpoints.sort()

    # constant definition
    height, width, channels = image.shape
    print(height)
    horizontal_cluster_size = int((height / row) / 8)
    vertical_cluster_size = int((width / column) / 5)
    endpoint_cluster_size = int((height / row) / 6)
    horizontal_merger_constant = int((width / column) / 5)
    upperShiftConstant = int(height/row/8)

    # order value translation:
    # 0 - x1,
    # 1 - y1,
    # 2 - x2,
    # 3 - y2
    # horizontal groups
    clusters_hori = grouper(horizontal, horizontal_cluster_size, 1)
    keys_hori = []
    [keys_hori.append(int(sum(x) / len(x))) for x in clusters_hori]

    # vertical groups:
    clusters_vert = grouper(vertical, vertical_cluster_size, 0)
    keys_vert = []
    [keys_vert.append(int(sum(x) / len(x))) for x in clusters_vert]

    # endpoints groups
    clusters_end = []
    for y in vertical_endpoints:

        if len(clusters_end) == 0:
            clusters_end.append([int(y)])
        else:
            total = 0
            for x in clusters_end[len(clusters_end)-1]:
                total+=x
            avg = total / len(clusters_end[len(clusters_end)-1])
            if int(y) - avg <= endpoint_cluster_size:
                clusters_end[len(clusters_end)-1].append(int(y))
            else:
                clusters_end.append([int(y)])
    keys_end = []
    for item in clusters_end:
        if not len(item) <= 8: 
            total = 0
            for x in item:
                total+=x
            keys_end.append(int(total/len(item)))

    # combine endpoints and horizontal lines to more robustly find horizontal lines
    keys_horizontal = []
    counter_end = 0
    counter_hori = 0
    while counter_end < len(keys_end) and counter_hori < len(keys_hori):
        if keys_end[counter_end] - keys_hori[counter_hori] > horizontal_merger_constant:
            keys_horizontal.append(keys_hori[counter_hori])
            counter_hori += 1
        elif keys_hori[counter_hori] - keys_end[counter_end]> horizontal_merger_constant:
            keys_horizontal.append(keys_end[counter_end])
            counter_end += 1
        else:
            keys_horizontal.append(keys_hori[counter_hori])
            counter_hori += 1
            counter_end += 1
    if counter_end < len(keys_end):
        for x in range(counter_end, len(keys_end)):
            keys_horizontal.append(keys_end[x])
    if counter_hori < len(keys_hori):
        for x in range(counter_hori, len(keys_hori)):
            keys_horizontal.append(keys_hori[x])

    # upper horizontal line shit to account for 3D cars
    for index in range(len(keys_horizontal)):
        if index % 2 == 0:
            if keys_horizontal[index] >= upperShiftConstant:
                keys_horizontal[index] -= upperShiftConstant
            else:
                keys_horizontal[index] = 0

    # save segmented pictures
    # if not os.path.exists("./segment"):
    #     os.mkdir("./segment")
    # for y in range(len(keys_vert)-1):
    #     for x in range(0, len(keys_horizontal)-1, 2):
    #         cv2.imwrite("./segment/"+"{:04d}".format(x/2) + "_" + "{:04d}".format(y) + '.jpg', \
    #                 image[keys_horizontal[x]-buffer: keys_horizontal[x+1]+buffer, keys_vert[y]-buffer:keys_vert[y+1]]+buffer)

    np.save("../setupData/vertical.npy", keys_vert)
    np.save("../setupData/horizontal.npy", keys_horizontal)
    print("vert: ")
    print(keys_vert)
    print("horizontal: ")
    print(keys_horizontal)
    print("hori: ")
    print(keys_hori)
    print("end: ")
    print(keys_end)

    # graph lines onto a picture
    grid = np.copy(image)
    for x in keys_vert:
        cv2.line(grid,(x,0),(x,height),(255,0,0),3)
    for x in keys_horizontal:
        cv2.line(grid,(0,x),(width,x),(0,255,0),3)
    
    


    ################################## display and save ############################
    # show pictures
    # cv2.imshow("Original", image)
    # cv2.imshow("Masked", mask)
    # cv2.imshow("Gray", gray)
    # cv2.imshow("Masked Grayscale", gray_mask)
    # cv2.imshow('Guas', gaus)
    # cv2.imshow("Canny", canny)
    # cv2.imshow("Hough", houghPic)
    # cv2.imshow("Horizontal", horizPic)
    # cv2.imshow("Vertical", vertPic)
    cv2.imshow("Grid", grid)

    cv2.waitKey(0)