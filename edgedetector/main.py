import argparse
import numpy as np
import cv2

parse = argparse.ArgumentParser(description="Simple edge detection")
parse.add_argument('image', help='image path')
args = parse.parse_args()

threshold = 20
img = cv2.imread(args.image)
h, w, _ = img.shape
new = np.zeros((h, w, 3), np.uint8)

for i in range(len(img)):
    for j in range(len(img[i])):
        v = 0
        m = (int(img[i][j][0]) + int(img[i][j][1]) + int(img[i][j][2])) / 3
        if(j > 0): l = (int(img[i][j-1][0]) + int(img[i][j-1][1]) + int(img[i][j-1][2])) / 3
        else: l = m
        if(i > 0): t = (int(img[i-1][j][0]) + int(img[i-1][j][1]) + int(img[i-1][j][2])) / 3
        else: t = m
        if(abs(m - t) + abs(m - l) >= threshold): new[i][j] = (255,255,255)


while(1):
    cv2.imshow('original', img)
    cv2.imshow('edges', new)
    if cv2.waitKey(20) & 0xFF ==27:
        break

cv2.destroyAllWindows()
