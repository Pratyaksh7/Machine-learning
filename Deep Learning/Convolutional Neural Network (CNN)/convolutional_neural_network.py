# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

# Part1 - Data Preprocessing
# Preprocessing the training set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Part2 - Building the CNN
# Initializing the CNN

# Step1 - Convolution

# Step2 - Pooling

# Adding a second Convolution Layer

# Step3 - Flattening

# Step4 - Full Connection

# Step5 - Output Layer
