import tensorflow as tf

# Import TensorFlow Datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#------------load()---------------#
# Loading dataset returns dataset and metadata
#-------dataset['train']----------#
# Gets train_dataset 60'000 images
#-------dataset['test']-----------#
# Gets test_dataset 10'000 images
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

#-------features['label']---------#
# create array with all names of label
# T-shirt/top, Trouser, Pullover etc.
class_names = metadata.features['label'].names
print("Class names: {}".format(class_names))

#---splits['train'].num_examples--#
# Returns total number of images of datasets
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

#---------normalize---------------#
# Function to convert grayscale values
# From 0-255 to 0-1
def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels

#---------map()-------------------#
# The map function applies the normalize function to each element 
# in the train and test datasets
train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)

#---------cache()-----------------#
# The first time you use the dataset, the images will be loaded from disk
# Caching will keep them in memory, making training faster
train_dataset =  train_dataset.cache()
test_dataset  =  test_dataset.cache()

#----------reshape()--------------#
# Take a single image, and remove the color dimension by reshaping
# RGB to Greyscale 
# Before (0,0,0) to (0)
for image, label in test_dataset.take(1):
  break
image = image.numpy().reshape((28,28))

#---------imshow()----------------#
# Plot the image - voila a piece of fashion clothing
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

#----------take(25)---------------#
# Show 25 images including corresponding class_name
plt.figure(figsize=(10,10))
for i, (image, label) in enumerate(train_dataset.take(25)):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
plt.show()

#------tf.keras.layers.Flatten-----#
# Input layer
# This layer transforms the images from a 2d-array of 28 × 28 pixels,
# to a 1d-array of 784 pixels (28*28). 
# Think of this layer as unstacking rows of pixels in the image and lining them up. 
# This layer has no parameters to learn, as it only reformats the data.
#------tf.keras.layers.Dense-------#
# Hidden Layer
# A densely connected layer of 128 neurons.
# Each neuron (or node) takes input from all 784 nodes in the previous layer,
# weighting that input according to hidden parameters which will be learned during training, 
# and outputs a single value to the next layer.
#------tf.keras.layers.Dense--------#
# Output Layer
# A 128-neuron, followed by 10-node softmax layer. 
# Each node represents a class of clothing. 
# As in the previous layer, 
# the final layer takes input from the 128 nodes in the layer before it, 
# and outputs a value in the range [0, 1], 
# representing the probability that the image belongs to that class. 
# The sum of all 10 node values is 1.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

#----------loss function------------#
# An algorithm for measuring how far the model's outputs are from the desired output. 
# The goal of training is this measures loss.
#---------Optimizer-----------------#
# An algorithm for adjusting the inner parameters of the model in order to minimize loss.
#---------Metrics-------------------#
# Used to monitor the training and testing steps. 
# The following example uses accuracy, the fraction of the images that are correctly classified.
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

#---------math.-------------------#
#---.shuffle(num_train_examples)--#
# Randomizes the order so our model cannot learn anything from the order of the examples.
#-------.batch(BATCH_SIZE)--------#
#  Tells model.fit to use batches of 32 images and labels when updating the model variables.
#--------.repeat()----------------#
# As soon as all the entries are read from the dataset and you try to read the next element, 
# the dataset will throw an error. 
# That's where .repeat() comes into play. 
# It will re-initialize the dataset.
BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

#---------epochs------------------#
# The epochs=5 parameter limits training to 5 full iterations of the training dataset, 
# so a total of 5 * 60000 = 300000 examples.
model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

#-----------evaluate()------------#
# Next, compare how the model performs on the test dataset. 
# Use all examples we have in the test dataset to assess accuracy.
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)

# predictions = 1 Batch of test_dataset
for test_images, test_labels in test_dataset.take(1):
  test_images = test_images.numpy()
  test_labels = test_labels.numpy()
  predictions = model.predict(test_images)

#(32, 10)
# Batchsize = 32
# Count labels = 10
print("predictions.shape", predictions.shape)

# [1.2084112e-05 1.4346701e-06 3.0577585e-02 4.7825029e-06 9.5519811e-01
# 1.2731636e-09 1.4196743e-02 1.5562900e-09 9.2185337e-06 1.6875600e-08]
# prediction 9.5519811e-01 (class 4 with 95% accuracy)
print("predictions[0]", predictions[0])

#---------np.argmax()------------#
# returns index with largest value
print("np.argmax(predictions[0]))", np.argmax(predictions[0]))

# Check if test_label[0] == np.argmax(predictions[0])
# When true prediction
print("test_labels[0]", test_labels[0])

def plot_image(i, predictions_array, true_labels, images):
  predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img[...,0], cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# Grab an image from the test dataset
img = test_images[0]

# (28, 28, 1)
print("img.shape: ", img.shape)

# Add the image to a batch where it's the only member.
img = np.array([img])

# (1, 28, 28, 1)
print("img.shape: ", img.shape)

# [[3.7481523e-06 2.6535437e-07 3.8001675e-03 5.4414668e-06 9.8617178e-01
# 1.3018811e-11 1.0014037e-02 2.0536749e-10 4.4483377e-06 1.4533903e-10]]
predictions_single = model.predict(img)
print("prediction_single: ", predictions_single)

#plots all labels with accuracy prediction_single 
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

#index class_name of prediction_single 
np.argmax(predictions_single[0])
print("np.argmax(predictions_single[0]): ", np.argmax(predictions_single[0]))