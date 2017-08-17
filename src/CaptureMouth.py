"""
Created by xusheng on 17/8/17

"""
#import numpy as np
import cv2
from cv2 import *
import os
from datetime import datetime

pathclassifier = os.getcwd() + '/'
haar_face = pathclassifier + '../haar/haarcascade_frontalface_default.xml'
haar_eyes = pathclassifier + '../haar/haarcascade_eye.xml'
haar_mouth = pathclassifier + '../haar/haarcascade_mcs_mouth.xml'

if not os.path.isfile(haar_face):
    raise ValueError('Haar cascade not found!\n' + os.getcwd())

def saveimg(folder, img, size1=48, size2=48):
    targetFolder = pathclassifier + folder + '/'
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)
    dim = (size1, size2)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    filename = targetFolder + 'img' + datetime.now().strftime('%m%d%H%M%S%f') + '.jpg'
    print ("Image Save :  {0}".format(filename))
    cv2.imwrite(filename, resized)
    return

def captureImageMouth(folder):
    faceDet = cv2.CascadeClassifier(haar_face)
    eyeDet = cv2.CascadeClassifier(haar_eyes)
    mouthDet = cv2.CascadeClassifier(haar_mouth)

    noOfEyes = 0
    while noOfEyes != 2:
        cap = cv2.VideoCapture(0)
        ret, img = cap.read()
        cap.release()
        # cv2.destroyAllWindows()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDet.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eyeDet.detectMultiScale(roi_gray)
            noOfEyes = len(eyes)
            if (noOfEyes) == 2:
                ex1, ey1, ew1, eh1 = eyes[0]
                ex1b, ey1b, ew1b, eh1b = eyes[1]
                eye1_gray = roi_gray[ey1:ey1 + eh1, ex1:ex1 + ew1]
                eye2_gray = roi_gray[ey1b:ey1b + eh1b, ex1b:ex1b + ew1b]
                saveimg(folder, eye1_gray, ew1, eh1)
                saveimg(folder, eye2_gray, ew1b, eh1b)
                if ex1b < ex1:
                    ex1 = ex1b
                dx, dy, dw, dh = x + ex1, y + ey1 + int(eh1 * 1.62), x + w - ex1, y + h
                roi_mouth = cv2.cvtColor(img[dy:dy + dh, dx:dx + dw], cv2.COLOR_BGR2GRAY)
                mouths = mouthDet.detectMultiScale(roi_mouth, 1.3, 5)
                if (len(mouths) > 0):
                    mx, my, mw, mh = mouths[0]
                    mouth_gray = roi_mouth[my:my + mh, mx:mx + mw]
                    saveimg(folder, mouth_gray, mw, mh)
    return


captureImageMouth('../image/test')