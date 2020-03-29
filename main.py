#IMPORTS
import cv2
import numpy as np
import matplotlib.pyplot as plt
from modules import plate_detect


## Scanning pic to RGB
img = cv2.imread('car_plate.jpg')
if img is None:
    print('failed to upload pic')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#DETECT PLATE
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
#using off the shelf haarcascade classifiers


detected_img ,x,y,width,height= plate_detect(img, plate_cascade)
#see plate_detect documntation

#plotting detected plate
plt.subplot(1,2,1)
plt.imshow(detected_img)
plt.title('step 1 - detect licence plate using Cascade Classifier ')




#BLUR PLATE
ROI = img[y:y+height,x:x+width]
ROI = cv2.medianBlur(ROI,7)
#Will use Median blur with K=3 = lox pixel image

#Create blurred pic
blurred_licenced_plate = img.copy()
blurred_licenced_plate[y:y+height,x:x+width] = ROI

#plot blurred pic
plt.subplot(1,2,2)
plt.imshow(blurred_licenced_plate)
plt.title('step 2 - Median Blurred licenced plate ')


plt.show()