
import numpy as np
import cv2

################################# functions ####################################

# detects whether or not points in ordered_list are groups
def grouper(ordered_list, cluster_size, order):
    points = []
    for line in ordered_list:
        y = line[0][order]
        if len(points) == 0:
            points.append([y])
        else:
            sum = 0
            for x in points[len(points)-1]:
                sum+=x
            avg = sum / len(points[len(points)-1])
            if y - avg <= cluster_size:
                points[len(points)-1].append(y)
            else:
                points.append([y])
    return points


#helper function for the quicksort function
def partition(arr,low,high, index): 
    i = ( low-1 )         # index of smaller element 
    pivot = arr[high][0][index]     # pivot 
  
    for j in range(low , high): 
  
        # If current element is smaller than or 
        # equal to pivot 
        if   arr[j][0][index] <= pivot: 
          
            # increment index of smaller element 
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i] 
  
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    return ( i+1 ) 
  
# Function to do Quick sort
def quickSort(arr,low,high,index): 
    if low < high: 
        pi = partition(arr,low,high,index) 
        quickSort(arr, low, pi-1, index) 
        quickSort(arr, pi+1, high, index) 

def rgb_mask(image):
    im = np.copy(image)
    height, width, channels = im.shape

    # left white color mask
    white_left_lower = np.array([130,118,118], dtype=np.uint8)
    white_left_upper = np.array([255,150,255], dtype=np.uint8)
    white_left_mask = cv2.inRange(im, white_left_lower, white_left_upper)
    white_left_mask[0:height, int(width/2)+30:width] = False

    # right white color mask
    white_right_lower = np.array([110,110,110], dtype=np.uint8)
    white_right_upper = np.array([255,255,255], dtype=np.uint8)
    white_right_mask = cv2.inRange(im, white_right_lower, white_right_upper)
    white_right_mask[0:height, 0:int(width/2)+30] = False

    # combine white masks
    white_mask = cv2.bitwise_or(white_left_mask, white_right_mask)

    # green color mask
    green_lower = np.array([0,0,0], dtype=np.uint8)
    green_upper = np.array([70,255,255], dtype=np.uint8)
    green_mask = cv2.inRange(im, green_lower, green_upper)

    # combine the green and white masks
    mask = cv2.bitwise_or(white_mask, green_mask)

    # apply and return masks
    masked = cv2.bitwise_and(im, im, mask = mask)
    return masked