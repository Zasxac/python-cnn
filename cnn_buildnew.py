import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import keras
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

epochs_val = 20

path = "C:\\Users\\monke\\Downloads\\python cnn\\"
    
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory(path + 'train19',
                                                 target_size = (128, 128),
                                                 batch_size = 64,
                                                 class_mode = 'categorical')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(path + 'test19',
                                            target_size = (128, 128),
                                            batch_size = 64,
                                            class_mode = 'categorical')
print("Image Processing.......Compleated")


#DISPLAY A BAR CHART SHOWING NO OF SAMPLES FOR EACH CATEGORY
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
    cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", activation='relu', input_shape=[128, 128, 3]))
    cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2,padding="same"))
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2,padding="same"))
    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2,padding="same"))
    cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2,padding="same"))
    cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation='relu'))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=19, activation='softmax'))
    print("Training cnn.....")
    cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return cnn 

cnn = myModel()    
print(cnn.summary())

buildModel = cnn.fit(training_set, validation_data = test_set, batch_size=64, epochs = epochs_val)

##### Save Model 
print("Creating Weight.....")
#path = 'modelbuild19_30e.h5'
#cnn.save(path)
print("Weight Created")

##### Hiện thị ma trận hỗn loạn để đánh giá độ chính xác
rebuildModel=keras.models.load_model('modelbuild19_30e.h5')

ans = rebuildModel.predict(test_set)
y = np.concatenate([test_set.next()[1] for i in range(test_set.__len__())])
y_pred = np.round(ans)

labels9 = ['Apple','Avocado','Banana','Cherry','Kiwi','Orange','Pineapple','Strawberries','Watermelon']
labels19 = [
    'Apple',
    'Avocado',
    'Banana',
    'Bell Pepper',
    'Bitter Gourd',
    'Blueberry',
    'Cherry',
    'Coconut',
    'Dragonfruit',
    'Durian',
    'Eggplant',
    'Grape',
    'Kiwi',
    'Lime',
    'Mangosteen',
    'Orange',
    'Peanut',
    'Tomato',
    'Watermelon'
    ]


cm = confusion_matrix(y_true=y.argmax(axis=1),y_pred=y_pred.argmax(axis=1),normalize='true')
display = ConfusionMatrixDisplay(cm,display_labels=labels19)
#display = ConfusionMatrixDisplay(cm,display_labels=labels9)
#display.plot(cmap=plt.cm.Blues)
fig, ax = plt.subplots(figsize=(16,9))
display.plot(ax=ax,xticks_rotation=45)
plt.show()

##### Plot đồ thị chỉ sổ
score =rebuildModel.evaluate(test_set,verbose=0)
print('Test Loss:',score[0])
print('Test Accuracy:',score[1])

fig,axes = plt.subplots(1,2, figsize=(15,8))
fig.suptitle("The model 's evaluation ",fontsize=20)

axes[0].plot(rebuildModel.history['loss'])
axes[0].plot(rebuildModel.history['val_loss'])
axes[0].set_title('Model Loss')
axes[0].legend(['Train','Test'])
axes[0].set_ylabel('Loss')
axes[0].set_xlabel('Epoch')
axes[0].grid()

axes[1].plot(rebuildModel.history['accuracy'])
axes[1].plot(rebuildModel.history['val_accuracy'])
axes[1].set_title('Model Accuracy')
axes[1].set_ylabel('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train','Test'])
axes[1].grid()
plt.show()

cv2.waitKey(0)