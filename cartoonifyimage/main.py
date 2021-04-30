import cv2 as cv
import numpy as np
import easygui

filetype = ["*.jpg", "*.png"]
f1 = easygui.fileopenbox()
img1 = cv.imread(f1)
grey = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
thresh1 = cv.adaptiveThreshold(grey, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                               cv.THRESH_BINARY, 9, 15)
# 5 param is block size from pixel, 6 param is constant C (threshold)
blur = cv.bilateralFilter(img1,9,300,300)
# 2 param is diameter from pixel, 3 is maximum difference to be mixed
result = cv.bitwise_and(blur, blur, mask=thresh1)
cv.namedWindow('original')
cv.namedWindow('cartoon')

while(1):
    cv.imshow('original', img1)
    cv.imshow('cartoon', result)
    if cv.waitKey(20) & 0xFF ==27:
        break
