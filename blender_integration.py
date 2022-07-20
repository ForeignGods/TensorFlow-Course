import bpy
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model, callbacks
from tensorflow.python.training import py_checkpoint_reader

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import math
import numpy as np
from numpy import array

from os.path import isfile
from pathlib import Path

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

############################################################
# Reset Data
############################################################

def reset_data():

    for ob in bpy.data.objects:   
        bpy.data.objects.remove(ob)
        
    for material in bpy.data.materials:
            bpy.data.materials.remove(material, do_unlink=True)

    for cur in bpy.data.curves:
        bpy.data.curves.remove(cur)
        
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)

############################################################
# Training
############################################################
        
def training():

    dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']

    class_names = metadata.features['label'].names

    num_train_examples = metadata.splits['train'].num_examples
    num_test_examples = metadata.splits['test'].num_examples

    def normalize(images, labels):
      images = tf.cast(images, tf.float32)
      images /= 255
      return images, labels

    train_dataset =  train_dataset.map(normalize)
    test_dataset  =  test_dataset.map(normalize)

    train_dataset =  train_dataset.cache()
    test_dataset  =  test_dataset.cache()

    mc = tf.keras.callbacks.ModelCheckpoint(filepath='weights.{epoch:02d}', save_weights_only=True, verbose=1, save_freq='epoch', )
    prediction_mc = tf.keras.callbacks.ModelCheckpoint(filepath='peights.{epoch:02d}', save_weights_only=True, verbose=1, save_freq=0, )
    #C:\Program Files\Blender Foundation\Blender 3.2 saves weights at this location
                                        
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(25, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    BATCH_SIZE = 32
    train_dataset = train_dataset.cache().shuffle(num_train_examples).batch(BATCH_SIZE)

    model.fit(train_dataset, epochs=5, callbacks=[mc], steps_per_epoch=num_train_examples/BATCH_SIZE)

    BATCH_SIZE = 1
    test_dataset = test_dataset.cache().batch(BATCH_SIZE)

    test_dataset = test_dataset.take(1000)

    train_images = []
    train_labels = []
    for train_image, train_label in test_dataset:
        train_image = train_image.numpy()
        train_label = train_label.numpy()
        train_images.append(train_image)
        train_labels.append(train_label)
    for img in train_images:
        print(img)
    for img in train_labels:
        print(img)


    model.fit(train_dataset, epochs=1000, callbacks=[prediction_mc], steps_per_epoch=1)

    weight_784_25_list = []
    weight_25_10_list = []
    path = 'C:\Program Files\Blender Foundation\Blender 3.2\peights.{zero}{index}'

    for i in range(1, 1000):
        if i <= 9:
            reader = py_checkpoint_reader.NewCheckpointReader(path.format(zero = 0, index = i))        
        else:
            reader = py_checkpoint_reader.NewCheckpointReader(path.format(zero = "", index = i))  

        state_dict = {
            v: reader.get_tensor(v) for v in reader.get_variable_to_shape_map()
        }
        weight_784_25_list.append(state_dict['layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE'])
        weight_25_10_list.append(state_dict['layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE'])

    return weight_784_25_list, weight_25_10_list, train_images, train_labels
   
############################################################
# Create Nodes
############################################################
    
def create_nodes():
    for row in range(28):
        for col in range(28):
            bpy.ops.mesh.primitive_cube_add(location = (row * 2, 0 , col * 2))
            bpy.context.active_object.name = 'input'.format(row).format(col)

    for row in range(5):
        for col in range(5):
            bpy.ops.mesh.primitive_ico_sphere_add(location = (row * 13.5 ,  -25, col * 13.5), scale=(0.5, 0.5, 0.5))
            bpy.context.active_object.name = 'hidden'.format(row).format(col)

    for row in range(10):
        bpy.ops.mesh.primitive_cube_add(location = (row * 6, -50, col * 6.75))
        bpy.context.active_object.name = 'output'.format(row).format(col)
    
    input_list_obj = []
    hidden_list_obj = []
    output_list_obj = []
    input_list_location = []
    hidden_list_location = []
    output_list_location = []

    for obj in bpy.data.objects:   
        if "input" in obj.name:
            input_list_obj.append(obj)
            input_list_location.append(obj.location)
        elif "hidden" in obj.name:
            hidden_list_obj.append(obj)
            hidden_list_location.append(obj.location)
        elif "output" in obj.name:
            output_list_obj.append(obj)
            output_list_location.append(obj.location)

    return input_list_obj, hidden_list_obj, output_list_obj, input_list_location, hidden_list_location, output_list_location
 
############################################################
# Create Multiple Single Connections
############################################################

def create_connection_single(start, end, name):
    coords = [start, end]
    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = 1
    curveData.bevel_depth = 0.0125

    # map coords to spline
    polyline = curveData.splines.new('POLY')
    polyline.points.add(len(coords)-1)
    for i, coord in enumerate(coords):
        x,y,z = coord
        polyline.points[i].co = (x, y, z, 1)

    # create Object
    curveOB = bpy.data.objects.new(name, curveData)

    # attach to scene and validate context
    #scn = bpy.context.scene
    #bpy.context.collection.objects.link(curveOB)
    
    return curveOB

############################################################
# Create Complex Connection Object
############################################################

def create_connection_complex(start_location_list, end_location_list, name, start_size, end_size):
    coords = []
    for x in range(start_size):
        for y in range(end_size):
            coords.extend((start_location_list[x], end_location_list[y]))

    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = 1
    curveData.extrude = 0.005

    # map coords to spline
    polyline = curveData.splines.new('POLY')
    polyline.points.add(len(coords)-1)
    for i, coord in enumerate(coords):
        x,y,z = coord
        polyline.points[i].co = (x, y, z, 1)

    # create Object
    curveOB = bpy.data.objects.new(name, curveData)

    #Set active object to variable
    #print(x,y)
    material = bpy.data.materials.new(name = "Basic")
    material.use_nodes = True
    bpy.context.object.active_material = material
    principled_node = material.node_tree.nodes.get("Principled BSDF")
    principled_node.inputs[19].default_value = (1,1,1,1)

    principled_node.inputs[20].default_value = 0.125
        
    curveOB.data.materials.append(material)
    # attach to scene and validate context
    scn = bpy.context.scene
    bpy.context.collection.objects.link(curveOB)

############################################################
# Assign material based on argmax weight values
############################################################

def single_assign_weight(name, start_location_list, end_location_list, weight_value_list, start_size, end_size):
    collection = bpy.context.blend_data.collections.new(name=name)
    bpy.context.collection.children.link(collection)
    for x in range(start_size):
        max_index = np.argmax(weight_value_list[x])
        for y in range(end_size):
            #print(weight_value_list[x][max_index]," == ",weight_value_list[x][y])
            if(weight_value_list[x][max_index] == weight_value_list[x][y] and weight_value_list[x][y] != 0 ):
                #print("connections created")
                curveOB = create_connection_single(start_location_list[x], end_location_list[y], name)
                material = bpy.data.materials.new(name = name)
                material.use_nodes = True
                bpy.context.object.active_material = material
                principled_node = material.node_tree.nodes.get("Principled BSDF")
                principled_node.inputs[19].default_value = (1,1,1,1)
                principled_node.inputs[20].default_value = 10
                curveOB.data.materials.append(material)
                collection.objects.link(curveOB)
                
############################################################
# Assign 28x28 materials Input
############################################################

def numpy_array_to_material(obj_list, labels_features, name):
    materials = []
    i=0
    if(name=="input"):
        pixels = labels_features.reshape(-1)
        for obj in obj_list:
            material = bpy.data.materials.new(name = name)
            material.use_nodes = True
            bpy.context.object.active_material = material
            principled_node = material.node_tree.nodes.get("Principled BSDF")
            principled_node.inputs[19].default_value = (1,1,1,1)
            principled_node.inputs[20].default_value = pixels[i]
            obj.data.materials.append(material)
            i += 1
    else:
        pixels = labels_features
        for obj in obj_list:
            if(labels_features == i):
                color = 1
            else:
                color = 0
            material = bpy.data.materials.new(name = name)
            material.use_nodes = True
            bpy.context.object.active_material = material
            principled_node = material.node_tree.nodes.get("Principled BSDF")
            principled_node.inputs[19].default_value = (1,1,1,1)
            principled_node.inputs[20].default_value = color
            obj.data.materials.append(material)
            i += 1
                
    return materials
                             
############################################################
# Main
############################################################

reset_data()

#weight_list contains 1000 lists of weights
weight_784_25_list, weight_25_10_list, train_images, train_labels = training()

input_list_obj, hidden_list_obj, output_list_obj, input_list_location, hidden_list_location, output_list_location = create_nodes()

material_list_input_hidden = single_assign_weight("input_hidden", input_list_location, hidden_list_location, weight_784_25_list[0], 784, 25)

material_list_hidden_output = single_assign_weight("hidden_output", hidden_list_location, output_list_location, weight_25_10_list[0], 25, 10)

create_connection_complex(input_list_location, hidden_list_location, "784_25_complex", 784, 25)

create_connection_complex(hidden_list_location, output_list_location, "25_10_complex", 25, 10)

input_materials = numpy_array_to_material(input_list_obj, train_images[6], "input")

output_materials = numpy_array_to_material(output_list_obj, train_labels[6], "output")

#def turn_on(view_layer: bpy.types.ViewLayer, collection_include: bpy.types.Collection):
#    for layer_collection in view_layer.layer_collection.children['ScriptAnimated'].children:
#        if layer_collection.collection == collection_include:
#            layer_collection.exclude = False
# 
#def turn_off(view_layer: bpy.types.ViewLayer, collection_include: bpy.types.Collection):
#    for layer_collection in view_layer.layer_collection.children['ScriptAnimated'].children:
#        if layer_collection.collection == collection_include:
#            layer_collection.exclude = True
# 
#def every_frame(scene):
#    print("Frame Change", scene.frame_current)
#    view_layer = bpy.context.scene.view_layers['View Layer']
#    
# #collection exclude animation
#    if scene.frame_current == 700:
#        turn_off(view_layer, bpy.data.collections["test"])
#    elif scene.frame_current == 1001: 
#        turn_on(view_layer, bpy.data.collections["test"])
# 
#bpy.app.handlers.frame_change_pre.append(every_frame)
