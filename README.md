# TensorFlow-Course
 Udacity Course "Intro to TensorFlow for Deep Learning"
## Glossary
<ul>
<li><strong>Feature:</strong> The input(s) to our model</li>
<li><strong>Examples:</strong> An input/output pair used for training</li>
<li><strong>Labels:</strong> The output of the model</li>
<li><strong>Layer:</strong> A collection of nodes connected together within a neural network.</li>
<li><strong>Model:</strong> The representation of your neural network</li>
<li><strong>Dense and Fully Connected (FC):</strong> Each node in one layer is connected to each node in the previous layer.</li>
<li><strong>Weights and biases:</strong> The internal variables of model</li>
<li><strong>Loss:</strong> The discrepancy between the desired output and the actual output</li>
<li><strong>MSE:</strong> Mean squared error, a type of loss function that counts a small number of large discrepancies as worse than a large number of small ones.</li>
<li><strong>Gradient Descent:</strong> An algorithm that changes the internal variables a bit at a time to gradually reduce the loss function.</li>
<li><strong>Optimizer:</strong> A specific implementation of the gradient descent algorithm. (There are many algorithms for this. In this course we will only use the “Adam” Optimizer, which stands for <em>ADAptive with Momentum</em>. It is considered the best-practice optimizer.)</li>
<li><strong>Learning rate:</strong> The “step size” for loss improvement during gradient descent.</li>
<li><strong>Batch:</strong> The set of examples used during training of the neural network</li>
<li><strong>Epoch:</strong> A full pass over the entire training dataset</li>
<li><strong>Forward pass:</strong> The computation of output values from input</li>
<li><strong>Backward pass (backpropagation):</strong> The calculation of internal variable adjustments according to the optimizer algorithm, starting from the output layer and working back through each layer to the input.</li>
</ul>
