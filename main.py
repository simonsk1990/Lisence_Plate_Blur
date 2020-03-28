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
plt.subplot(1,2,1)
detected_img = plate_detect(img, plate_cascade)
plt.imshow(detected_img)
# plt.imshow(plate_detect(img,plate_cascade))
plt.title('step 1 - detect licence plate ')

#BLUR PLATE





plt.show()