# Import dependencies 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

import os
import zipfile

#TF Version - Check to be the same as in requirments.txt
print(tf.__version__)

# Set Variables:
_BATCH_SIZE = 20
_EPOCHS = 100
IMG_HEIGHT = 224
IMG_WIDTH = 224

## When working on Google Colab
#Check current directory
!pwd

#Go to directory
os.chdir('/tmp')
!pwd

#Remove Folder
!rm -rf disney_picture_data

# Mount Google Drive
from google.colab import drive
drive.mount('/content/gdrive')

# Create a temp. directory where we gonna store the picture data
try: 
  os.mkdir('/tmp/disney_picture_data')
  # Alternatively use !MKDIR
except OSError:
    pass

# Go to directory
os.chdir('/tmp/')

# Unzip our .zip file in the directory
!unzip "{data path}" -d 'disney_picture_data'

# Declare path
PATH = '/tmp/disney_picture_data/'

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

# Data preparation
# Generator for our training data
train_image_generator = ImageDataGenerator(
      rescale=1./255,       
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest') 

# Generator for our validation data
validation_image_generator = ImageDataGenerator(
      rescale=1./255,       
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

#Run - Train generator
train_data_gen = train_image_generator.flow_from_directory(
    batch_size=_BATCH_SIZE,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')

# Label
labels = (train_data_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
print(labels)

# The next function returns a batch from the dataset
sample_training_images, _ = next(train_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
# Plot images
plotImages(sample_training_images[:5])

IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# Mobile Net model trainable = FALSE
base_model.trainable = False

# Base model architecture
base_model.summary()

# Callback 
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.95):
      print("\nReached 95% accuracy so cancelling training!")
      self.model.stop_training = True
      
# Model
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer2 = tf.keras.layers.Dense(units = 128, input_shape = (520,), activation='relu')
prediction_layer = tf.keras.layers.Dense(units = 14, input_shape = (128,), activation='softmax')

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer2,
  prediction_layer
])

# Model compilation
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Model summary
model.summary()

# Train model
history = model.fit(
    train_data_gen,
    steps_per_epoch=5,
    epochs=_EPOCHS,
    verbose=1,
    validation_data=val_data_gen,
    validation_steps=14,
    callbacks=[myCallback()]
)

# Training vizulaization
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(_EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('/tmp/disney_model.h5')

##########################################################
#Single Images Prediction:
##########################################################
# Define image path of our Prediction Image
image_path = "/content/gdrive/My Drive/_EPIC_Machine_Learning/_My Flask Apps/Disney Picture Material"

# Put files into lists and return them as one list with all images in the folder
def loadImages(path):

    image_file = sorted([os.path.join(path, file)
                          for file in os.listdir(path )
                          if file.endswith(('.jpg'))])
    return image_file

# Define image_list (note there is only one image)
image_list = loadImages(image_path)
path = np.array(image_list)
path_string = (path[11])
img = tf.io.read_file(path_string)
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.convert_image_dtype(img, tf.float32)
print(img.shape)

final_img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

plt.subplot(121), plt.imshow(final_img)
print(final_img.shape)

final_img_tfl = np.expand_dims(final_img, axis=0)
print(final_img_tfl.shape)

#Expand Tensor for Model (Input shape)
y = np.expand_dims(final_img, axis=0)

#Predict Image Tensor with model
prediction = model.predict(y)
prediction_squeeze = np.squeeze(prediction, axis=0)

label_array = np.array(labels)

#print(type(label))
for key, value in labels.items():
    real_label = prediction_squeeze[key]
    
    print ("{0:.0%}".format(real_label), value)
