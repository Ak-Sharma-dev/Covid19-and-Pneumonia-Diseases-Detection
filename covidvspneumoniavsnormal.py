# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 20:14:28 2020

@author: HP
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

covid=cv2.imread('train/COVID-19 cases/radiol.2020200490.fig3.jpeg')
covid=cv2.cvtColor(covid,cv2.COLOR_BGR2RGB)
plt.imshow(covid)
plt.title('COVID PERSON XRAY LUNG IMAGE') 

covid=cv2.imread('train/Pneumonia/person1671_virus_2887.jpeg')
covid=cv2.cvtColor(covid,cv2.COLOR_BGR2RGB)
plt.imshow(covid)
plt.title('PNEUMONIA PERSON XRAY LUNG IMAGE')

normal=cv2.imread('train/Normal/NORMAL2-IM-0370-0001.jpeg')
normal=cv2.cvtColor(normal,cv2.COLOR_BGR2RGB)
plt.imshow(normal)
plt.title('NORMAL PERSON XRAY LUNG IMAGE')

INPUT_W = 1200
INPUT_H = 900
DIVIDER = 3.6

#INPUT_DIM = (int(INPUT_W/DIVIDER),int(INPUT_H/DIVIDER),3)
#INPUT_ARRAY = (int(INPUT_W/DIVIDER),int(INPUT_H/DIVIDER))

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (5, 5), input_shape = (128 , 128, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (3, 3)))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (4, 4), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a fourth convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a fifth convolutional layer
classifier.add(Conv2D(512, (2, 2), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dense(units = 3, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.summary()
# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (128,128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('test',
                                            target_size = (128,128),
                                            batch_size = 32,
                                            class_mode = 'categorical')
                         
history=classifier.fit_generator(
    training_set,
    epochs=30,
    steps_per_epoch=30,
    validation_data=test_set) 
    
# Save the model
classifier.save('Models/ML_model1.h5')

import matplotlib.pyplot as plt
# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'],'r')
plt.plot(history.history['val_acc'],'g')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'],'r')
plt.plot(history.history['val_loss'],'g')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# predicting the output as normal or pneumonia patient on a single image
imag=cv2.imread('test/Normal/NORMAL2-IM-0274-0001.jpeg')
imag=cv2.cvtColor(imag,cv2.COLOR_BGR2RGB)
plt.imshow(imag)
plt.title('CLASSIFY THIS IMAGE')

from keras.preprocessing import image
test_image = image.load_img('test/Normal/NORMAL2-IM-0274-0001.jpeg', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1 :
    prediction = 'COVID-19 INFECTED PERSON';
elif result[0][1] == 1:
    prediction = 'NORMAL PERSON';
elif result[0][2] == 1:
    prediction = 'PNEUMONIA INFECTED PERSON';
print(prediction)

