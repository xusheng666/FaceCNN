"""
Created by xusheng on 14/8/17

"""
from keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np
import random
import sys
from Utility import convert_img_to_csv


# emotion labels from FER2013:
emotion = {'Angry': 0,  'Fear': 1, 'Happy': 2,
           'Sad': 3, 'Surprise': 4, 'Neutral': 5}
emo     = ['Angry', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']

def reconstruct(pix_str, size=(96,96)):
    pix_arr = np.array(map(int, pix_str.split()))
    print pix_arr

    return pix_arr.reshape(size)

def emotion_count(y_train, classes, verbose=True):
    emo_classcount = {}
    print 'Disgust classified as Angry'
    # y_train.loc[y_train == 1] = 0
    # classes.remove('Disgust')
    for new_num, _class in enumerate(classes):
        y_train.loc[(y_train == emotion[_class])] = new_num
        class_count = sum(y_train == (new_num))
        if verbose:
            print '{}: {} with {} samples'.format(new_num, _class, class_count)
        emo_classcount[_class] = (new_num, class_count)
    return y_train.values, emo_classcount

def load_data(sample_split=0.3, usage='Training', to_cat=True, verbose=True,
              classes=['Angry','Happy'], filepath='../data/training_96.csv'):
    df = pd.read_csv(filepath)
    # print df.tail()
    # print df.Usage.value_counts()
    # df = df[df.Usage == usage]
    frames = []
    # classes.append('Disgust')
    for _class in classes:
        class_df = df[df['emotion'] == emotion[_class]]
        frames.append(class_df)
    data = pd.concat(frames, axis=0)
    rows = random.sample(data.index, int(len(data)*sample_split))
    data = data.ix[rows]
    print '{} set for {}: {}'.format(usage, classes, data.shape)
    data['pixels'] = data.pixels.apply(lambda x: reconstruct(x))
    x = np.array([mat for mat in data.pixels]) # (n_samples, img_width, img_height)
    X_train = x.reshape(-1, 1, x.shape[1], x.shape[2])
    y_train, new_dict = emotion_count(data.emotion, classes, verbose)
    print new_dict
    if to_cat:
        y_train = to_categorical(y_train)
    return X_train, y_train, new_dict

def save_data(usage='', fname='', to_cat=True, folder='../data/'):
    emo = ['Angry', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']
    X_train, y_train, emo_dict = load_data(sample_split=1.0,
                                           classes=emo,
                                           usage=usage,
                                           to_cat=to_cat,
                                           verbose=True)
    np.save(folder + 'X_train' + fname, X_train)
    np.save(folder + 'y_train' + fname, y_train)
    print X_train.shape
    print y_train.shape

if __name__ == '__main__':
    # generate the train csv file for training
    print 'Prepare Training file...'
    convert_img_to_csv('../data/meme_faces_96/')

    # makes the numpy arrays ready to use:
    print 'Making moves...'
    # emo = ['Angry', 'Fear', 'Happy',
    #        'Sad', 'Surprise', 'Neutral']
    # X_train, y_train, emo_dict = load_data(sample_split=1.0,
    #                                        classes=emo,
    #                                        usage='PrivateTest',
    #                                        verbose=True)
    print 'Saving...'
    save_data('Training', fname='_md_self', to_cat=True)
    # save_data('PrivateTest', fname='_privatetest6_100pct')
    # save_data('PublicTest', fname='_publictest6_100pct')

    print 'Done!'
