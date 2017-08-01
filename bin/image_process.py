import numpy as np

def addnoise(image,percent =0):
    row, column = image.shape
    image_copy = image
    for i in range(row):
        for j in range(column):
            if np.random.uniform(0,100) < percent:
                image_copy[i,j] = int(not image[i,j])
    return image_copy

def half(image):
    row, column = image.shape
    image[0:row/2,:] = 0
    return image
