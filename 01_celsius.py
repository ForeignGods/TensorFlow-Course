import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for i,c in enumerate(celsius_q):
  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))
  

#------------input_shape-----------#
# This specifies that the input to this layer is a single value. 
# That is, the shape is a one-dimensional array with one member.
#------------units-----------------#
# The single value is a floating point number, 
# representing degrees Celsius.
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

#------------model-----------------#
# Once layers are defined, they need to be assembled into a model.
# The Sequential model definition takes a list of layers as an argument,
# specifying the calculation order from the input to the output.
# This model has just a single layer, called "l0".
model = tf.keras.Sequential([l0])

#------------loss function---------#
# A way of measuring how far off predictions are from the desired outcome. 
# (The measured difference is called the "loss".)
#------------optimizer function----#
# A way of adjusting internal values in order to reduce the loss.
#------------adam------------------#
# Learning rate (0.1)
# This is the step size taken when adjusting values in the model.
model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.1))

#------------fit()-----------------#
# The fit method returns a history object. 
# We can use this object to plot how the loss
# of our model goes down after each training epoch. 
# A high loss means that the Fahrenheit degrees the model predicts
# is far from the corresponding value in fahrenheit_a.
#------------arg1/input------------#
# Input values as array in this case celsius_q.
#------------arg2/output-----------#
# Output values as array in this case fahrenheit_a.
#------------arg3/epochs-----------#
# The epochs argument specifies how many times this cycle should be run.
#------------arg4/verbose----------#
# The verbose argument controls how much output the method produces.
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)

#------------matplotlib------------#
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()

#------------predict()-------------#
# You can use the predict method to have it calculate
# the Fahrenheit degrees for a previously unknown Celsius degrees.
# Remember, 100 Celsius was not part of our training data.
print(model.predict([100.0]))# Close plt tp print.

#------------l0.get_weights()------#
# Finally, let's print the internal variables of the Dense layer. 
# The first variable is close to ~1.8 and the second to ~32. 
# These values (1.8 and 32) are the actual variables in the real conversion formula.
# The conversion equation, f=1.8c+32.
print("These are the layer variables: {}".format(l0.get_weights()))
