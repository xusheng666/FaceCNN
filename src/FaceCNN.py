"""
Created by xusheng on 13/8/17

"""

import cv2
import os
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import plotly.plotly as py

from keras.models import model_from_json

pathclassifier = os.getcwd() + '/'

def saveimg(folder, img):
    dim = (48, 48)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    filename = pathclassifier + folder + 'img' + datetime.now().strftime('%Y%m%d_%H%M%S%f') + '.jpg'
    cv2.imwrite(filename, resized)

def load_model(modePath="../data/results"):
    with open(modePath+'/cnn_model.txt', 'r') as f:
        json_string = f.read()
        model = model_from_json(json_string)
        f.close
        return model

"""
API for predict by the real time from camera
"""
def predict_emotion(face_image_gray): # a single cropped face
    # load previous trained model
    model = load_model()
    model.load_weights("../data/weights/cnn_weights")

    resized_img = cv2.resize(face_image_gray, (48,48), interpolation = cv2.INTER_AREA)
    image = resized_img.reshape(1, 1, 48, 48)
    list_of_list = model.predict(image, batch_size=1, verbose=1)
    angry, fear, happy, sad, surprise, neutral = [prob for lst in list_of_list for prob in lst]
    return [angry, fear, happy, sad, surprise, neutral]

"""
API for predict by the read from the image files
"""
def predict_emotion_from_img(face_image_gray): # a single cropped face
    # load previous trained model
    model = load_model()
    model.load_weights("../data/weights/cnn_weights")
    newarray = face_image_gray.ravel()
    resized_img = cv2.resize(newarray, (48,48), interpolation = cv2.INTER_AREA)
    image = resized_img.reshape(1, 1, 48, 48)
    print image
    list_of_list = model.predict(image, batch_size=1, verbose=1)
    angry, fear, happy, sad, surprise, neutral = [prob for lst in list_of_list for prob in lst]
    return [angry, fear, happy, sad, surprise, neutral]

"""
Testing with read file API
"""
def read_file_detetion():
    filev = '/Users/xusheng/PycharmProjects/FaceCNN/data/meme_faces/angry/img20170812_161208530000.jpg'

    fileimg = cv2.imread(filev)
    probs = predict_emotion_from_img(fileimg)
    print probs

"""
Testing with real time camera
"""
def real_time_detection():

    cascPath = pathclassifier + 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)

    video_capture = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, 1)

        faces = faceCascade.detectMultiScale(
            img_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            face_image_gray = img_gray[y:y + h, x:x + w]
            print face_image_gray
            probs = predict_emotion(face_image_gray)
            # probsimg = predict_emotion(fileimg)

            saveimg("../data/meme_faces/", face_image_gray)
            print probs
            # print probsimg
            # xs = [1,2,3,4,5,6]
            # LABELS = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            # plt.bar(xs, probs)
            # plt.xticks(xs, LABELS)
            # fig = plt.gcf()
            # plot_url = py.plot_mpl(fig, filename='mpl-basic-bar')
            # break
        break
    video_capture.release()

if __name__ == '__main__':
    try:
        # real_time_detection()
        read_file_detetion()
    except:
        print "Unexpected error:",   sys.exc_info()[0]
        raise