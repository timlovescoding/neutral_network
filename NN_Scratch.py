"""
Building a neural network from scratch without any deep learning library. Only NumPy was used.
@author: Timothy Lim

*huge credits to Professor Andrew Ng Coursera Neural Network Course
*codes only to be used for education purposes

"""
# Libraries...

import numpy as np
import matplotlib.pyplot as plt


# STEP 1 : Initialization:....................................................................

def initialise_parameters(layer_dims): # initialising forward propagation values
    # Takes in the dimensions required for your neutral network (Layer Dimension)
    # layers_dims = ( size of features (n_x),  size of hidden layer 1 (n_h1),...(size of other n_h)...., size of output (n_y))

    parameters = {} # Dictionary Type, easy access for variable name using key
    L = len(layer_dims) # Number of layers wanted from neural network

    for i in range(1,L): #Note: Range does not take upper bound number (which is what we want as input layer is not part of the hidden layer)
        # Number of hidden layers = Number of weight and bias matrix needed

        parameters["W" + str(i)] =  np.random.randn(layers_dims[i], layers_dims[i-1]) / np.sqrt(layers_dims[i-1] )   # Weight Matrix
        parameters["B" + str(i)] =  np.zeros((layers_dims[i], 1))  # Bias Matrix (size: 1 column)

    return  parameters   # Returning parameters needed for forward propagation


#  STEP 2: Forward Propagation:........................................................................

def sigmoid(x):  # Sigmoid Function
  return  1 / (1 + np.exp(-x))


def forward_activation(A_prev, W , b , activation):

    Z = np.dot(W,A_prev) + b  # Z: Value to put into function (sigmoid/ReLu) to get next activation unit
    linear_cache = (A_prev,W,b) # Cache for backward propagation
    # A: Activation Unit, W: Weight, b: bias

    if activation == "sigmoid":
       A = sigmoid(Z) # next activation unit
       activation_cache  = Z

    elif activation == "relu":
        A = np.maximum(0,Z) # Activation through relu
        activation_cache = Z

    cache = (linear_cache, activation_cache)

    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    assert(A.shape == (W.shape[0], A_prev.shape[1]))

    return A,cache



def forward_propagate(X, parameters):

    caches = [] #Initialise empty cache (to append later on)
    A = X # initial Activation unit is the input features
    L = len(parameters) // 2 # Getting the length of hidden layer (Note: it is floor-ed cause indexes cannot take in float)

    for i in range(1,L): #remember the last weight is not included

        A_prev  =  A #Current Activation Unit value that was calculated (Starting with inputs)
        A,cache =  forward_activation(A_prev, parameters["W"+str(i)] , parameters["B"+str(i)] , "relu") # Using ReLu Activation
        caches.append(cache) # Add values of the variables into the list

    #For last activation: use sigmoid

    A_Last,cache =  forward_activation(A, parameters["W"+str(L)] , parameters["B"+str(L)] , "sigmoid") #calculating last activation unit
    caches.append(cache) # Add values of the variables into the list


    return A_Last, caches



# STEP 3: Cost Computation......................................................................

def compute_cost(A_Last, Y): # A_Last: Last activation unit (prediction) , Y: Data Output

     m  = Y.shape[1]  # Number of training samples ( Make sure output is size(1,Samples) )

     cost = -(1/m) * ( np.dot(Y, np.log(A_Last).T)  +  np.dot( (1-Y), np.log(1-A_Last).T) ) # Using logarithmic cost function

     cost = np.squeeze(cost) # ensure cost array of size (1,1) to become just a singular number
     # Squeeze is important as array multiplication in python always gives back an array

     assert(cost.shape == ())
     return cost


# STEP 4: Backward Propagation:.................................................................

def backward_linear(dZ , cache):

    A_prev, W, b = cache # From forward propogation
    m = A_prev.shape[1] # Column of input features/activation units =  number of samples

    # Backprop Gradient formula: Remember to try to derive them yourself!

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m *  np.sum(dZ, axis = 1, keepdims =True)
    dA_prev = np.dot(W.T,dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def backward_activation(dA , cache , activation):

    linear_cache, activation_cache = cache

    if activation == "relu":

        Z = activation_cache

        dZ = np.array(dA, copy=True) # Copy dA over to dZ
        #dA is the derivative of dZ when above 0, think of y = x, dA/dZ  = 1 (dA is y while dZ is x in this case)

        dZ[Z <= 0] = 0   #Logical index, when Z <= 0 , dZ will be = 0 (ReLU function)


        dA_prev, dW, db = backward_linear(dZ, linear_cache) # getting the gradient for linear calculations


    elif activation == "sigmoid":

        Z = activation_cache

        dZ =  dA  * sigmoid(Z)  * ( 1-sigmoid(Z) )  # Derivative for Sigmoid activation function (dZ)
        dA_prev, dW, db = backward_linear(dZ, linear_cache) # getting the gradient for linear calculations


    return dA_prev, dW, db


def backward_propagate(A_Last,Y,caches):

    gradients = {}
    L = len(caches) # Number of hidden layer
    Y = Y.reshape(A_Last.shape)  #Make sure outputs follows the same shape as the last activation unit

    # Backward propogate the output first (initialising) :

    dA_Last = -(np.divide(Y, A_Last) - np.divide(1 - Y, 1 - A_Last)) # Derivative derived from Cost Function
    current_cache = caches[L-1] # the last variable in the cache
    gradients["dA"+str(L-1)] , gradients["dW" + str(L)], gradients["dB" + str(L)] = backward_activation(dA_Last,current_cache,'sigmoid')

    for i in reversed(range(L-1)): #Going from the back of the range (from second last to 0), remember that the last one is already done (a step above)

        current_cache = caches[i]
        dA_prev_temp, dW_temp, db_temp =  backward_activation(gradients["dA" + str(i+1)],current_cache,'relu')
        gradients["dA" + str(i)] = dA_prev_temp
        gradients["dW" + str(i + 1)] = dW_temp
        gradients["dB" + str(i + 1)] = db_temp

    return gradients

 # STEP 5: Gradient Descent.............................................................................


def gradient_descent(parameters , gradients, learning_rate):

    L = len(parameters) // 2   #number of hidden layers

    for i in range(L):
        parameters["W" + str(i+1)] = parameters["W" + str(i+1)] - learning_rate * gradients["dW"+str(i+1)]
        parameters["B" + str(i+1)] = parameters["B" + str(i+1)] - learning_rate * gradients["dB"+str(i+1)]

    return parameters




# STEP 6: Putting it all together................................................................................

def NN_model(X,Y, layers_dims, learning_rate , iterations ):

    costs = [] # keeping track of cost (plot later or print to confirm that error is decreasing)

    parameters = initialise_parameters(layers_dims) # get all the parameters necessary according to the wanted NN layers.

    for i in range(0, iterations):

        A_Last, caches =  forward_propagate(X, parameters) #Forward Propagation

        cost = compute_cost(A_Last, Y) #print later

        gradients = backward_propagate(A_Last, Y, caches) #Backward Propagation

        parameters = gradient_descent(parameters , gradients, learning_rate)

        if  i % 100 == 0: # Every 100 iterations, add a point (cost)
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters # Weight and Bias Matrix that best fits the training data.

# Checking the accuracy of your AI: Good luck!.........................................................

def predict(X,Y, parameters): # Predicting based off NN predictions

     m = X.shape[1] # Note: The column of the input matrix is the number of samples
     #h_l = len(parameters) // 2 #number of hidden layers
     prediction = np.zeros((1,m)) # because the output matrix is size (1,number of samples)

     # Forward Propagate the Updated Parameters

     A_Last , caches = forward_propagate(X, parameters)  #A_Last is the prediction

     for i in range(0,m):
         if A_Last[0,i] > 0.5: # using sigmoid, if probability above 0.5 == positive
             prediction[0,i] = 1
         else:
             prediction[0,i] = 0

     #Check Accuracy    :

     accuracy = np.sum((prediction == Y)/m)

     print("Accuracy: "  + str(accuracy) )

     return prediction


'''
 How to use the neutral network algorithm:

 Step 1. Set up input data (X) to be size - (number of features, number of samples)
 Step 2. Set up output data (Y) to be size - (1,number of samples) as NN returns output of size (1 , number of samples)
 Step 3. Set up the size and number of neutral network layers that you want to test with.
         layers_dims = ( size of input features (n_x),  size of hidden layer 1 (n_h1),...(size of other n_h)...., size of output (n_y))
 Step 4. Use the NN function. NN_model(X,Y, layers_dims, learning_rate , iterations )
        - Cost function will be plotted. Use it to make sure cost is reaching steady-state.
        - Tune Learning rate accordingly
 Step 5. Use the predict function. Accuracy of prediction will be computed.
       - predict(train_x, train_y, parameters). Updated parameters comes from NN_model
       - Use on Training Data first. ( Ensure it is of high accuracy to begin with)
       - Proceed with Testing Data.
       - Accuracy of testing data is usually pretty low with this NN. We can definitely improve it!
       - Stay tune for updates!

'''

# THINGS YOU HAVE TO DO :........................................................

# (1) Set your inputs  as train_x, split some for test_x  (Flatten the image into 1D data)
# (2) Set the labels as train_y, split some for test_y
# (3) Have fun! Try different amount of activation units and hidden layers. Try tuning the learning rate and iterations too!

'''Example of code once you have your training and testing data set up.
layers_dims = (12288,30,15,8, 1) # Neural Network
parameters =  NN_model(train_x,train_y, layers_dims, learning_rate = 0.0080, iterations = 2000 )
accuracy_train = predict(test_x, test_y, parameters)
'''


