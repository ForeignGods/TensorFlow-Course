import bpy
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.python.training import py_checkpoint_reader

# Import TensorFlow Datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Helper libraries
import math
import numpy as np
from numpy import array

from os.path import isfile
from pathlib import Path

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#delete every existing object
for ob in bpy.data.objects:   
    bpy.data.objects.remove(ob)
    
for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

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

#---splits['train'].num_examples--#
# Returns total number of images of datasets
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

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

mc = tf.keras.callbacks.ModelCheckpoint('C:\Temp', save_weights_only=True, period=1)
                                     
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
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
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

model.fit(train_dataset, epochs=5, callbacks=[mc], steps_per_epoch=num_train_examples/BATCH_SIZE)

#print(mc)
#print(callback_custom.layer_1_list)

#-----------evaluate()------------#
# Next, compare how the model performs on the test dataset. 
# Use all examples we have in the test dataset to assess accuracy.
#test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
#print('Accuracy on test dataset:', test_accuracy)

reader = py_checkpoint_reader.NewCheckpointReader('C:\Temp')

state_dict = {
    v: reader.get_tensor(v) for v in reader.get_variable_to_shape_map()
}

print(state_dict.keys())
print(type(state_dict))
array_100 = state_dict['layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE']
#print(array_784_100)
        
#print(dir(bpy.context.active_object))#get all attributes

#print(array_784_100.shape)

for key in state_dict:
    obj = reader.get_tensor(key)
    if isinstance(obj, np.ndarray):
        print("tensor_name: ", key)
        print("type:  ", type(reader.get_tensor(key)))
        #print("np.ndarray:  ", reader.get_tensor(key))
        print("np.ndarray:  ", reader.get_tensor(key).shape)

#for row in range(28):
#    for col in range(28):
#        bpy.ops.mesh.primitive_cube_add(location = (row * 2, col * 2, 0))
#        bpy.context.active_object.name = 'input'.format(row).format(col)

materials = []
for row in range(10):
    for col in range(10):
        bpy.ops.mesh.primitive_ico_sphere_add(location = (row * 6 ,col * 6 , -25))
        bpy.context.active_object.name = 'hidden'.format(row).format(col)
        activeObject = bpy.context.active_object #Set active object to variable
        material = bpy.data.materials.new(name = "Basic")
        material.use_nodes = True
        bpy.context.object.active_material = material
        principled_node = material.node_tree.nodes.get("Principled BSDF")
        principled_node.inputs[19].default_value = (1,1,1,1)
        principled_node.inputs[20].default_value = 10
        activeObject.data.materials.append(material)
        materials.append(material)
        
i = 0
for x in array_100:

    principled_node = materials[i].node_tree.nodes.get("Principled BSDF")
    principled_node.inputs[20].default_value = x
    
    i += 1

#for row in range(10):
#    bpy.ops.mesh.primitive_cube_add(location = (row * 6, col * 6 - 27, -50))# +27= ofset to center
#    bpy.context.active_object.name = 'output'.format(row).format(col)
#    
#    
#input_list_obj = []
#hidden_list_obj = []
#output_list_obj = []
#input_list_location = []
#hidden_list_location = []
#output_list_location = []

#for obj in bpy.data.objects:   
#    if "input" in obj.name:
#        input_list_obj.append(obj)
#        input_list_location.append(obj.location)
#    elif "hidden" in obj.name:
#        hidden_list_obj.append(obj)
#        hidden_list_location.append(obj.location)
#    elif "output" in obj.name:
#        output_list_obj.append(obj)
#        output_list_location.append(obj.location)

#for cur in bpy.data.curves:
#    bpy.data.curves.remove(cur)

#coords = []
#for x in range(784):
#    for y in range(100):
#        #print(x,y)
#        coords.extend((input_list_location[x],hidden_list_location[y]))
#        
#for x in range(100):
#    for y in range(10):
#        #print(x,y)
#        coords.extend((hidden_list_location[x],output_list_location[y]))


#curveData = bpy.data.curves.new('myCurve', type='CURVE')
#curveData.dimensions = '3D'
#curveData.resolution_u = 1

## map coords to spline
#polyline = curveData.splines.new('POLY')
#polyline.points.add(len(coords)-1)
#for i, coord in enumerate(coords):
#    x,y,z = coord
#    polyline.points[i].co = (x, y, z, 1)

## create Object
#curveOB = bpy.data.objects.new('myCurve', curveData)
##curveData.bevel_depth = 0.01

## attach to scene and validate context
#scn = bpy.context.scene
#bpy.context.collection.objects.link(curveOB)
## create the Curve Datablock
