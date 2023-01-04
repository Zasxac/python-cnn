import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
import threading
import queue
from keras.preprocessing.image import ImageDataGenerator

i = 0  
font = cv2.FONT_HERSHEY_SIMPLEX
pathfinal = 'final'

cnn = keras.models.load_model('modelcnn.h5') 

def getName9(classNo):
    if   classNo == 0: return 'Apple'
    elif classNo == 1: return 'Avocado'
    elif classNo == 2: return 'Banana'
    elif classNo == 3: return 'Cherry'
    elif classNo == 4: return 'Kiwi'
    elif classNo == 5: return 'Orange'
    elif classNo == 6: return 'Pineapple'
    elif classNo == 7: return 'Strawberries'
    elif classNo == 8: return 'Watermelon'
    else: pass

def getName19(classNo):
    if   classNo == 0: return 'Apple'
    elif classNo == 1: return 'Avocado'
    elif classNo == 2: return 'Banana'
    elif classNo == 3: return 'Bell Pepper'
    elif classNo == 4: return 'Bitter Gourd'
    elif classNo == 5: return 'Blueberry'
    elif classNo == 6: return 'Cherry'
    elif classNo == 7: return 'Coconut'
    elif classNo == 8: return 'Dragonfruit'
    elif classNo == 9: return 'Durian'
    elif classNo == 10: return 'Eggplant'
    elif classNo == 11: return 'Grape'
    elif classNo == 12: return 'Kiwi'
    elif classNo == 13: return 'Lime'
    elif classNo == 14: return 'Mangosteen'
    elif classNo == 15: return 'Orange'
    elif classNo == 16: return 'Peanut'
    elif classNo == 17: return 'Tomato'
    elif classNo == 18: return 'Watermelon'
    else: pass
    
def getClass(classNo):
    if   (classNo == 0 or
          classNo == 1 or 
          classNo == 2 or 
          classNo == 5 or 
          classNo == 6 or 
          classNo == 7 or 
          classNo == 8 or
          classNo == 9 or 
          classNo == 11 or 
          classNo == 12 or
          classNo == 13 or 
          classNo == 14 or 
          classNo == 15 or 
          classNo == 18
          ) : return 'Fruits'
    elif (classNo == 3 or 
          classNo == 4 or 
          classNo == 10 or 
          classNo == 16 or 
          classNo == 17
          ) : return 'Vegetables'
    else: pass
    
vid = cv2.VideoCapture(0)
#vid = cv2.VideoCapture('http://192.168.1.101:8081/video')

print("Camera connection successfully")

while(True):  
    r, frame = vid.read() 
    frame=cv2.flip(frame,1)
    cv2.putText(frame, "Name:" , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Class:" , (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(pathfinal+str(i)+".jpg", frame)
    test_image = tf.keras.utils.load_img(pathfinal+str(i)+".jpg", target_size = (128, 128))
    test_image = tf.keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image)
    
    classIndex = np.argmax(result,axis=1)
    
    cv2.putText(frame, str(getName19(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame,str(getClass(classIndex)) , (120, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Show your fruit', frame)
    os.remove(pathfinal+str(i)+".jpg")
    i = i + 1
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
vid.release() 
cv2.destroyAllWindows() 


