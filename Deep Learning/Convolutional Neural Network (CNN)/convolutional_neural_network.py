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
cnn = tf.keras.models.Sequential()

# Step1 - Convolution
cnn.add(tf.keras.layers.Conv2D(kernel_size=3, filters=32, activation='relu', input_shape=[64, 64, 3]))

# Step2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second Convolution Layer
cnn.add(tf.keras.layers.Conv2D(kernel_size=3, filters=32, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# Part3 - Training the CNN
# Compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN on the training set and evaluating it on the Test Set
cnn.fit(x= training_set, validation_data=test_set, epochs=25)

# Part4 - Making a single prediction
import numpy as np
from keras_preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image =np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)

# The actual image was of a dog
# Prediction -> dog

# The training and test dataset has not been uploaded because of file size issues


# epochs accuracy
# Epoch 1/25
# 250/250 [==============================] - ETA: 0s - loss: 0.6726 - accuracy: 0.5778
# 250/250 [==============================] - 205s 821ms/step - loss: 0.6726 - accuracy: 0.5778 - val_loss: 0.6244 - val_accuracy: 0.6690
# Epoch 2/25
# 250/250 [==============================] - 38s 150ms/step - loss: 0.6078 - accuracy: 0.6706 - val_loss: 0.5597 - val_accuracy: 0.7155
# Epoch 3/25
# 250/250 [==============================] - 37s 147ms/step - loss: 0.5565 - accuracy: 0.7143 - val_loss: 0.5178 - val_accuracy: 0.7550
# Epoch 4/25
# 250/250 [==============================] - 41s 165ms/step - loss: 0.5177 - accuracy: 0.7401 - val_loss: 0.4987 - val_accuracy: 0.7585
# Epoch 5/25
# 250/250 [==============================] - 49s 194ms/step - loss: 0.5007 - accuracy: 0.7470 - val_loss: 0.5080 - val_accuracy: 0.7650
# Epoch 6/25
# 250/250 [==============================] - 54s 215ms/step - loss: 0.4776 - accuracy: 0.7724 - val_loss: 0.4872 - val_accuracy: 0.7600
# Epoch 7/25
# 250/250 [==============================] - 35s 141ms/step - loss: 0.4584 - accuracy: 0.7785 - val_loss: 0.4731 - val_accuracy: 0.7890
# Epoch 8/25
# 250/250 [==============================] - 39s 155ms/step - loss: 0.4422 - accuracy: 0.7893 - val_loss: 0.5635 - val_accuracy: 0.7360
# Epoch 9/25
# 250/250 [==============================] - 48s 190ms/step - loss: 0.4341 - accuracy: 0.7956 - val_loss: 0.4528 - val_accuracy: 0.8000
# Epoch 10/25
# 250/250 [==============================] - 55s 219ms/step - loss: 0.4149 - accuracy: 0.8125 - val_loss: 0.4659 - val_accuracy: 0.7800
# Epoch 11/25
# 250/250 [==============================] - 49s 197ms/step - loss: 0.3995 - accuracy: 0.8161 - val_loss: 0.4517 - val_accuracy: 0.8055
# Epoch 12/25
# 250/250 [==============================] - 37s 149ms/step - loss: 0.3890 - accuracy: 0.8284 - val_loss: 0.4602 - val_accuracy: 0.7940
# Epoch 13/25
# 250/250 [==============================] - 42s 166ms/step - loss: 0.3727 - accuracy: 0.8306 - val_loss: 0.4591 - val_accuracy: 0.7995
# Epoch 14/25
# 250/250 [==============================] - 57s 228ms/step - loss: 0.3570 - accuracy: 0.8438 - val_loss: 0.4388 - val_accuracy: 0.8090
# Epoch 15/25
# 250/250 [==============================] - 48s 192ms/step - loss: 0.3390 - accuracy: 0.8482 - val_loss: 0.4518 - val_accuracy: 0.8050
# Epoch 16/25
# 250/250 [==============================] - 56s 222ms/step - loss: 0.3202 - accuracy: 0.8555 - val_loss: 0.5356 - val_accuracy: 0.7775
# Epoch 17/25
# 250/250 [==============================] - 55s 222ms/step - loss: 0.3102 - accuracy: 0.8673 - val_loss: 0.5012 - val_accuracy: 0.7955
# Epoch 18/25
# 250/250 [==============================] - 57s 227ms/step - loss: 0.2932 - accuracy: 0.8729 - val_loss: 0.4842 - val_accuracy: 0.7940
# Epoch 19/25
# 250/250 [==============================] - 58s 231ms/step - loss: 0.2785 - accuracy: 0.8830 - val_loss: 0.5048 - val_accuracy: 0.7910
# Epoch 20/25
# 250/250 [==============================] - 53s 211ms/step - loss: 0.2621 - accuracy: 0.8949 - val_loss: 0.5868 - val_accuracy: 0.7760
# Epoch 21/25
# 250/250 [==============================] - 52s 207ms/step - loss: 0.2553 - accuracy: 0.8950 - val_loss: 0.5083 - val_accuracy: 0.8010
# Epoch 22/25
# 250/250 [==============================] - 56s 223ms/step - loss: 0.2307 - accuracy: 0.9064 - val_loss: 0.5253 - val_accuracy: 0.8065
# Epoch 23/25
# 250/250 [==============================] - 56s 225ms/step - loss: 0.2218 - accuracy: 0.9086 - val_loss: 0.5390 - val_accuracy: 0.8080
# Epoch 24/25
# 250/250 [==============================] - 55s 222ms/step - loss: 0.2108 - accuracy: 0.9180 - val_loss: 0.5311 - val_accuracy: 0.8040
# Epoch 25/25
# 250/250 [==============================] - 57s 229ms/step - loss: 0.1962 - accuracy: 0.9205 - val_loss: 0.5937 - val_accuracy: 0.7850