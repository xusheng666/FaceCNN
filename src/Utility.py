# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 2017

@author: Abner
"""
import os
import cv2
import sys
import csv

dictEmotion = {'angry': '0', 'fear': '1', 'smile': '2', 'sad': '3','surprised': '4','neutral': '5'}
listEmotion = ['angry','fear', 'smile', 'sad','surprised','neutral']

def convert_img_to_csv(rootdir='../data/results/'):
    # read all the images and write to a csv with emtion, pixel
    with open('../data/training.csv', 'wb') as csvfile:
        csvfile.write("emotion, pixels")
        csvfile.write('\r')

        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                fileName = os.path.join(subdir, file)
                # print fileName
                if fileName.endswith('.jpg'):
                    array = fileName.rsplit('/')
                    # print fileName
                    # print array[array.__len__()-2]
                    if listEmotion.__contains__(array[array.__len__()-2]):
                        emotion = dictEmotion[array[array.__len__()-2]]
                        # print emotion
                        fileimg = cv2.imread(fileName)
                        one1array = convert_img_array_to_1d_array(fileimg)
                        arrayText = " ".join([str(x) for x in one1array])
                        # print one1array
                        rowText = emotion + "," + arrayText
                        print rowText
                        csvfile.write(rowText+"\r")
        csvfile.close()

def convert_img_array_to_1d_array(rbgArray):
    # return the 1 d array of the input 3 d RBG array
    onedarray = rbgArray.ravel()
    return onedarray


if __name__ == '__main__':
    try:
        # real_time_detection()
        convert_img_to_csv('../data/meme_faces/')
    except:
        print "Unexpected error:",   sys.exc_info()[0]
        raise