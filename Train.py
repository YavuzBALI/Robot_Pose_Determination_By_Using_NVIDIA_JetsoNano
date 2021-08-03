import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np 
import os

#Purged the GPU from previous work.
tf.keras.backend.clear_session()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
TF_GPU_ALLOCATOR='cuda_malloc_async'

#Scaled the dataset between 0-1 in order to be able to
process faster.
train=ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)

#Pulled the dataset from the files
train_dataset=train.flow_from_directory('Data/trainData/',target_size=(101,101),batch_size=3,class_mode='categorical')
Validation_dataset=validation.flow_from_directory('Data/ValidationData/',target_size=(101,101),batch_size=3,class_mode='categorical')

#Printed class, class induces and image size for datasets
print(train_dataset.class_indices)
print(train_dataset.classes)
print(train_dataset.image_shape)

#Here built a sequential model with the help of keras
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(20,(5,5),strides=(1,1),padding='same',input_shape =(101,101,3)),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Activation("relu"),
                                    tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
                                    #
                                    tf.keras.layers.Conv2D(30,(11,11),strides=(1,1),padding='same'),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Activation("relu"),
                                    tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
                                    #
                                    tf.keras.layers.Conv2D(10,(7,7),strides=(1,1),padding='same'),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Activation("relu"),
                                    #
                                    tf.keras.layers.Flatten(),
                
                                    #
                                    tf.keras.layers.Dense(72),
                                    tf.keras.layers.Activation("softmax")
                                    ])
#Configures the model for training.
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
#We print the structure of the model to the screen
model.summary()

#Recorded the most successful weights
checkpointer = tf.keras.callbacks.ModelCheckpoint('Egit_agirliklari/Robot_Angle_pre3.h5',verbose=1,save_best_only=True)
#The section where the trainig takes place
hist = model.fit(train_dataset,batch_size=8,epochs=18,shuffle=True,validation_data=Validation_dataset,callbacks=[checkpointer],verbose=1)

#Saved in SavedModel format
model.save('Egit_agirliklari/Robot_Angle_pre_SavedModel_101')

'''
#save model H5 format to json
model_json = model.to_json()
with open("Egit_agirliklari/Robot_angle_pre.json","w") as json_file:
    json_file.write(model_json)
'''

#Graphically expressing the training and validation results obtained as a result of the training#and printing them on the screen
#In this part,we draw the loss graph.
plt.figure(figsize=(14,3))
plt.subplot(1,2,1)
plt.suptitle('Train',fontsize=10)
plt.ylabel("Loss",fontsize= 16)
plt.plot(hist.history['loss'],color='b',label='training_loss')
plt.plot(hist.history['val_loss'],color='r',label= 'validation_loss')
plt.legend(loc='upper right')

#In this part, drawn the accuracy graph
plt.subplot(1,2,2)
plt.ylabel("Accuracy",fontsize= 16)
plt.plot(hist.history['accuracy'],color='b',label='training_accuracy')
plt.plot(hist.history['val_accuracy'],color='r',label= 'validation_Accuracuy')
plt.legend(loc='lower right')
plt.show()
