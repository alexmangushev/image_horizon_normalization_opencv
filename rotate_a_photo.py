import cv2
import math
import numpy as np

img = cv2.imread('books/books_4_1.jpg')
copy_img = img
img = cv2.resize(img, (512, 512 * img.shape[0] // img.shape[1]))

# convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# use bilateral filter
#blur = cv2.bilateralFilter(gray, 5, 80, 5)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# use sobel to finds max of abs(grad)
grad_x = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# erode
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(sobel, kernel, iterations = 1)

# use canny to make bin mask of edge
canny = cv2.Canny(erosion, 100, 180, None, 3)

#contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

'''
# find all your connected components (white blobs in your image)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(canny, connectivity = 8)

# function yields every seperated component with information on each of them, such as size
# the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
#sizes = stats[1:, -1]
#nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)
# here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
#min_size = 15

# your answer image
img2 = np.zeros((output.shape))
# for every component in the image, you keep it only if it's above min_size

for i in range(nb_components):

    #con = cv2.minAreaRect(contours[i])
    #area = con[1][0] * con[1][1]

    #if sizes[i] >= min_size:
    if (max(stats[i][2], stats[i][3]) / min(stats[i][2], stats[i][3])) >= 1.5 and max(stats[i][2], stats[i][3]) >= 5.0:
        img2[output == i + 1] = 255

'''

lines = cv2.HoughLines(canny, 1, np.pi / 180, 120, None, 0, 0)

max_count = 0
ind = 0
if lines is not None:
    for i in range(0, len(lines)):
        theta = lines[i][0][1]
        kol = 0
        for k in range(i + 1, len(lines)):
            theta1 = lines[k][0][1]
            if (abs(theta1 - theta) <= 0.01):
                kol+=1
        if (kol > max_count):
            max_count = kol
            ind = i


rotation = 180 * lines[ind][0][1] / math.pi
if (rotation > 90):
    rotation = int(abs(90 - rotation))
else:
    rotation = int(-abs(90 - rotation))
#print(rotation)

# Draw the lines
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(img, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)


rho = lines[ind][0][0]
theta = lines[ind][0][1]
a = math.cos(theta)
b = math.sin(theta)
x0 = a * rho
y0 = b * rho
pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
cv2.line(img, pt1, pt2, (255, 0, 0), 3, cv2.LINE_AA)

rows, cols, _ = copy_img.shape
M = cv2.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), rotation, 1)
rez = cv2.warpAffine(copy_img, M, (cols, rows))

cv2.imshow('IMG', img)
#cv2.imshow('sobel', sobel)
#cv2.imshow('erode', canny)
cv2.imshow('rez', rez)

cv2.waitKey(0)