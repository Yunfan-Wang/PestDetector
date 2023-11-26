import os
import tensorflow as tf
import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import numpy as np
from keras import applications
#BATCH_SIZE = 64
"""""
train_generator = ImageDataGenerator(rotation_range=90, 
                                     brightness_range=[0.1, 0.7],
                                     width_shift_range=0.5, 
                                     height_shift_range=0.5,
                                     horizontal_flip=True, 
                                     vertical_flip=True,
                                     validation_split=0.15,
                                     preprocessing_function=preprocess_input) # VGG16 preprocessing

test_generator = ImageDataGenerator(preprocessing_function=preprocess_input) # VGG16 preprocessing
"""""
data_dir = Path('../Dataset/train')
print(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg'))) + len(list(data_dir.glob('*/*.png'))) 
print(image_count)
train_dir = Path('../Dataset/train')
val_dir = Path('../Dataset/test')
"""""
batch_size = 32
img_height = 244
img_width = 244

"""""
image_size = (224, 224)
batch_size = 32
train_datagen = ImageDataGenerator(rescale = 1./255,
                            shear_range = 0.4,        #Shear Intensity
                            zoom_range = 0.4,         #Range for random zoom
                            horizontal_flip = True,   #Randomly flip inputs horizontally. 
                            validation_split = 0.15,  #Fraction of images reserved for validation
                            rotation_range=20,        #Degree range for random rotations.    
                            width_shift_range=0.2,
                            height_shift_range=0.2)
train_ds = train_datagen.flow_from_directory(train_dir,
                                      target_size = image_size,
                                      batch_size = batch_size,
                                      class_mode = 'categorical',
                                      subset = 'training',
                                      color_mode="rgb")
val_ds = train_datagen.flow_from_directory(val_dir,
                                      target_size = image_size,
                                      batch_size = batch_size,
                                      class_mode = 'categorical',
                                      subset = 'validation',
                                      color_mode="rgb")



def create_model(input_shape, n_classes, optimizer='rmsprop'):
    """
    Compiles a model integrated with VGG16 pretrained layers

    
    input_shape: tuple - the shape of input images (width, height, channels)
    n_classes: int - number of classes for the output layer
    optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
    fine_tune: int - The number of pre-trained layers to unfreeze.
                If set to 0, all pretrained layers will freeze during training
    """
    
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = keras.applications.MobileNetV2(include_top=True,
                     weights='imagenet', 
                     input_shape=input_shape)
    
    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.

    # Create a new 'top' of the model (i.e. fully-connected layers).
    # This is 'bootstrapping' a new top_model onto the pretrained layers.
    
    #top_model = Flatten(name="flatten")(top_model)
    #top_model = Dropout(0.2)(top_model)
    top_model = conv_base.layers[-2].output
    top_model = Dense(128, activation='relu')(top_model)
    #top_model = Dropout(0.2)(top_model)
    top_model = Dense(64, activation='relu')(top_model)
    #top_model = Dropout(0.2)(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)
    
    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=conv_base.input, outputs=output_layer)

    # Compiles the model for training.
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    for i, layer in enumerate (model.layers):
       print(i, layer.name, "-", layer.trainable)

    return model

"""""
Early Stopping
"""""

# ModelCheckpoint callback - save best weights
tl_checkpoint_1 = ModelCheckpoint(filepath='tl_model_v1.weights.best.hdf5',
                                  save_best_only=True,
                                  verbose=1)

# EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',
                           patience=5,
                           restore_best_weights=True,
                           mode='min')
input_shape = (224, 224, 3)
optim_1 ="sgd"#keras.optimizers.RMSprop(lr=1e-5)#Adam(learning_rate=0.0001) 
n_classes=3

n_epochs = 10

"""""
C&F
"""""

# First we'll train the model without Fine-tuning
# For
MobNet = create_model(input_shape, n_classes, optim_1)


Res_history = MobNet.fit(train_ds,
                            batch_size=32,
                            epochs=10,
                            validation_data=val_ds,
                            callbacks=[tl_checkpoint_1, early_stop],
                            verbose=1)


MobNet.save("mobnet.h5")






















