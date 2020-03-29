#IMPORTS


import cv2
import numpy as np


def plate_detect(img,cascade_classifier):
    """
    :param img: src image
    :param cascade_classifier: var under cv2.CascadeClassifier
    :return: tuple[0] = img with rectangle over the features
    :return: tuple[1] tuple[2]= x,y top left coordinates
    :return: tuple[3] tuple[4]= width height
    """
    img_copy = np.copy(img)
    plate_cascade = cascade_classifier

    parameters = plate_cascade.detectMultiScale(img_copy, scaleFactor=1.2, minNeighbors=5)  #
    # receive x,y ,width, high
    for (x, y, width, height) in parameters:
        cv2.rectangle(img_copy, (x, y), (x + width, y + height), (255), 5)
    return img_copy, parameters[0,0], parameters[0,1],parameters[0,2],parameters[0,3]