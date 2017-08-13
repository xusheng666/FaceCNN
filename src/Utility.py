# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 2017

@author: Abner
"""

def convert_img_to_csv(dirpath='../data/results/'):
    # read all the images and write to a csv with emtion, pixel
    with open(dirpath + starttime +'.txt', 'r') as f:
        f.write(json_string)

def convert_img_array_to_1d_array(rbgArray):
    # return the 1 d array of the input 3 d RBG array
    onedarray = rbgArray.ravel()
    return onedarray