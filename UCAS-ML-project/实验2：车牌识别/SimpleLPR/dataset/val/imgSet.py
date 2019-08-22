# encoding=utf8

import cv2
import sys
import os


# province
for i in range(0,31):
    dir = './province/%s/' % i
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            filename1 = dir + filename
            img = cv2.imread(filename1)
            img = cv2.resize(img, (20, 20))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            filename = dir + filename.split(".")[0]+".jpg"
            os.remove(filename1)
            cv2.imwrite(filename, img)
            

# area
for i in range(0, 26):
    dir = './area/%s/' % i
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            filename1 = dir + filename
            img = cv2.imread(filename1)
            img = cv2.resize(img, (20, 20))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            filename = dir + filename.split(".")[0]+".jpg"
            os.remove(filename1)
            cv2.imwrite(filename, img)

# letter
for i in range(0, 34):
    dir = './letter/%s/' % i
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            filename1 = dir + filename
            img = cv2.imread(filename1)
            img = cv2.resize(img, (20, 20))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            filename = dir + filename.split(".")[0]+".jpg"
            os.remove(filename1)
            cv2.imwrite(filename, img)
        
