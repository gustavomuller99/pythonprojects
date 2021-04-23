import argparse
import pandas as pd
import cv2

def getColor(rgb):
    minimum = 1e9
    r = []
    for k in range(len(df)):
        c = [df.loc[k,'R'], df.loc[k,'G'], df.loc[k,'B']]
        value = abs(c[0] - rgb[0]) + abs(c[1] - rgb[1]) + abs(c[2] - rgb[2])
        if(value < minimum):
            minimum = value
            r = [c, df.loc[k, 'prettyname'], df.loc[k, 'hex']]
    return r

def isDark(rgb):
    return (rgb[0] + rgb[1] + rgb[2]) < 100

def mouseOver(event, x, y, flags, param):
    if(event == cv2.EVENT_MOUSEMOVE):
        r = getColor([img[y,x,0], img[y,x,1], img[y,x,2]])
        h, w, _ = img.shape
        if(isDark(r[0])): cv2.rectangle(img,(10,10),(w-10,40),(255,255,255),-1)
        else: cv2.rectangle(img,(10,10),(w-10,40),(0,0,0),-1)
        text = "Name: {};  HEX: {}".format(r[1], r[2])
        cv2.putText(img,text,(15,32),cv2.FONT_HERSHEY_SIMPLEX,0.7,(int(r[0][0]), int(r[0][1]), int(r[0][2])),2)


parse = argparse.ArgumentParser(description="Simple color detection")
parse.add_argument('image', help='image path')
args = parse.parse_args()

index = ["name", "prettyname", "hex", "R", "G", "B"]
df = pd.read_csv('colors.csv', names=index)

img = cv2.imread(args.image)
img = cv2.copyMakeBorder(img,50,0,0,0,cv2.BORDER_CONSTANT)
cv2.namedWindow('colordetector')
cv2.setMouseCallback('colordetector', mouseOver)

while(1):
    cv2.imshow('colordetector', img)
    if cv2.waitKey(20) & 0xFF ==27:
        break

cv2.destroyAllWindows()
