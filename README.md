# Building a neural network from Scratch (Using only NumPy)

The purpose of this project is to gain a strong foundation on Neural Networks (NN) which are the building blocks of Artificial Intelligence. By coding Neural Networks from scratch, a better understanding of the intricacies of NN will be gained. High level API such as Tensorflow and PyTorch will then be easily understood when needed to be use and further research to improve established frameworks can be made.

In this document, I will try my very best to run through every part of NN and also point you to the implementation in codes.

## Introduction to Neural Network (NN)

The human brain consist of billions of interconnected neurons receiving and transmitting information, providing us with the capability to deal with complex and demanding tasks. Artificial Neural Network (ANN) was developed to emulate the learning abilities of biological neurons system. The learning abilities of ANN made it an useful solution for a variety of complex problems such as facial recognition, speech recognition, image classification, machine translation which are used to be tasks that only humans can do but made possible for machine by utilising ANN. The main question for ANN is how does it learn to solve these tasks? In order to understand that, we will need to look at the overall architecture of the network and understand all of the functions used within the network.

## Neural Network Architecture

As shown in the image, the NN architecture consist of the **input, hidden and output layers**. All of these layers consist of multiple nodes which are connected to other nodes. Each nodes will consist of numeric value. Note that each layer for NN has 1D vector of nodes so any 2D and above data will be required to be flatten/reshape into the 1D space of numbers to be fed into the NN. The amount of nodes in the input layer follows the input data. Hidden layers are the layers connecting the input layer to the output layer. We have the flexibility to choose as many hidden layers as we want and vary the amount of nodes in each hidden layers as we see fit. The output layer nodes consist of values which we used to compared to the ground truth of our data for learning. This section provides you with more questions than answers about NN which is intended. Let's keep going by asking questions!

#### How are the nodes connected (what are the black lines)?

A node is connected to another node by a very simple and familiar equation (Y = MX + C). In the case of NN, the notations for the equation is **Y = Wx + b**  where  Y = Output, W= Weights, x=Input, b= Bias. Value of the node is multiplied by the weight value and then added with the bias value to produce a output value (Y) to be used for the connecting node. **Refer to code: , **

An issue with just using this equation (Y=Wx+b) is that it is only a linear equation which means that the NN unable to learn non-linear representation. NN is appealing in the first place because of the ability to extract information in non-linear patterns and **Tim: add universal approxmiation with non linear here?**.

To resolve this issue, we will take the node output value (Y) and put it into an **activation function** which gives us f(Y) for f is the activation function. There are many activation functions that we can look into but let's put our focus on the Sigmoid and ReLu function as it is most popularly used.


I found a visualization to aid the explanation, please head to the author's article [HERE](https://towardsdatascience.com/forward-propagation-in-neural-networks-simplified-math-and-code-version-bbcfef6f9250or more information) for more information if needed:

**add gif**


-Show pic of sigmoid and relu
**Research more into activation functions, who thought of it and how it came about**



The values for the weights (W) and biases (b) of the connections can be initialized through a few different methods:

**Research more than just xavier((

The network of moving and computing values from one node to the next node in another layer in a forward direction only is called a **Feedforward Network**. The process of computing f(Wx+b) is known as forward propagation as values are computed and propagated forward to reach the output nodes.




## Cost Function


 2. Cost function
 3. Backpropogation (Computing equations of gradients)
 4. Gradient Descent 
 5. Optimization

