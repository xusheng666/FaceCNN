# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 13:53:37 2017

@author: Earl
"""

import numpy as np
import cv2
import os
from datetime import datetime

pathclassifier = os.getcwd() + '/'
haar1 = pathclassifier + 'haarcascade_frontalface_default.xml'


def saveimg(folder, img):
    dim = (48, 48)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    filename = pathclassifier + folder + 'img' + datetime.now().strftime('%Y%m%d_%H%M%S%f') + '.jpg'
    cv2.imwrite(filename, resized)
    return


if not os.path.isfile(haar1):
    raise ValueError('Haar cascade not found!\n' + os.getcwd())


def captureImage(folder):
    faceDet = cv2.CascadeClassifier(haar1)

    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    cap.release()
    # cv2.destroyAllWindows()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDet.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        saveimg(folder, roi_gray)


captureImage('image/surprised/')