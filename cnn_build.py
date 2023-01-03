import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator

epochs_val = 25

path = "C:\\Users\\monke\\Downloads\\python cnn\\"
    
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory(path + 'train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(path + 'test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
print("Image Processing.......Compleated")

############################### DISPLAY A BAR CHART SHOWING NO OF SAMPLES FOR EACH CATEGORY
#print()
#plt.figure(figsize=(12, 4))
#plt.bar(range(0, ), num_of_samples)
#plt.title("Distribution of the training dataset")
#plt.xlabel("Class number")
#plt.ylabel("Number of images")
#plt.show()

def myModel():
    cnn = tf.keras.models.Sequential()
    print("Building Neural Network.....")
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return cnn

cnn = myModel()
print(cnn.summary())
history = cnn.fit(x = training_set, validation_data = test_set, epochs = epochs_val)

############################### PLOT
score =cnn.evaluate(test_set,verbose=0)
print('Test Score:',score[0])
print('Test Accuracy:',score[1])

fig,axes = plt.subplots(1,2, figsize=(15,8))
fig.suptitle("The model 's evaluation ",fontsize=20)

axes[0].plot(history.history['loss'])
axes[0].plot(history.history['val_loss'])
axes[0].set_title('Model Loss')
axes[0].legend(['Train','Test'])
axes[0].set_ylabel('Loss')
axes[0].set_xlabel('Epoch')

axes[1].plot(history.history['accuracy'])
axes[1].plot(history.history['val_accuracy'])
axes[1].set_title('Model Accuracy')
axes[1].set_ylabel('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train','Test'])
plt.show()

############################### Save Model 
print("Creating Weight...")
path = 'modelbuild.h5'
cnn.save(path)
print("Weight Created")
loaded_model= tf.keras.models.load_model(path)
cv2.waitKey(0)