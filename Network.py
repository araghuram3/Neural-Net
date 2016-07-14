"""
This is the code for my basic Neural Network. It is just something simple that I felt like doing one day. 
It is useful to test out different neural network structures on a training data set and visualize the error to determine which
structure works best for your dataset.
In order to use this, you'll need Python 2.7, the numpy module and the matplotlib.pyplot module if you want to use the plotter class.

@author Ankit Raghuram (ankit.raghuram.36@gmail.com)
@version 1.0

TODO implement a cost function instead of plotting error.
TODO incorporate "momentum" later.
TODO update the bias variable on iterations.
TODO include more activation functions for trial purposes.
"""
import numpy as np
import math


# Activation functions are defined below
def sigmoid(val):
    return 1/(1+math.exp(-val))

def sigDeriv(val):
    return sigmoid(val) * (1 - sigmoid(val))

class Network:
    """
    This is the network class. It takes in number of inputs, hidden layer structure, 
    activation function, and a learning rate. If nothing is provided, the the network
    will default with the parameters listed below.

    Values are stored for visualization purposes so you can see the progression of the
    numbers in the hidden layers. These are before the activation function is used.

    The "layer" class variables are really the weights of the layers. These are the things
    that are being stored and changed. 

    Feed Forward and Back Propagation aren't specifically used outside, but are used within
    methods that should be called outside (train, test).

    This implementation, while can handle many types of input and hidden layer structures,
    only supports one output variable for the time being. 
    """
    def __init__(self, inputs=2, hiddens=[3], activationFunction=sigmoid, derivative=sigDeriv, learningRate=.5, bias=1):

        # pick random weights initially for the inputs
        numLayers = len(hiddens) + 1
        self.weights = [np.array(0) for x in xrange(numLayers)]
        self.weights[0] = np.random.rand(hiddens[0], inputs)

        # put hidden layers into the layers list
        self.nodeValues = [np.array(0) for x in xrange(numLayers+1)]      # use for backpropogation (has sum values, not AF values)
        self.activationValues = [np.array(0) for x in xrange(numLayers)]  # has activation values also for back prop
        for x in xrange(1,len(hiddens)+1):
            num_nodes = hiddens[x-1]

            # check to see the length of the next layer
            # if there isn't a next one, use the output layer (1)
            self.weights[x] = np.random.rand(1, num_nodes) if x == len(hiddens) else np.random.rand(hiddens[x], num_nodes)
            
        # initialize list of error to track progression
        self.error = []

        # store the acivation function, learning rate and bias
        self.AF = np.vectorize(activationFunction)
        self.dAF = np.vectorize(derivative)
        self.LR = learningRate
        self.bias = bias

    def feedForward(self, inputData, verbose=False):

        # place inputData into nodeValues
        self.nodeValues[0] = inputData

        # find the first hidden layers values from inputs
        node_sum = np.dot(self.weights[0], inputData)
        self.nodeValues[1] = node_sum
        af_sum = self.AF(node_sum)
        self.activationValues[0] = af_sum

        if verbose:
            print '============================================='
            print 'i->h1 weights * inputs = h1 ==> activation h1'
            print self.weights[0]
            print '*'
            print inputData
            print '='
            print node_sum
            print '==>'
            print af_sum
            print '============================================='

        # iterate through the hidden layers to get the value for the output
        for x in xrange(1,len(self.weights)):
            hiddenLayer = self.weights[x]
            node_sum = np.dot(hiddenLayer, af_sum) + self.bias
            temp_af_sum = af_sum        # this is for verbose
            self.nodeValues[x+1] = node_sum
            af_sum = self.AF(node_sum)
            self.activationValues[x] = af_sum
            if verbose:
                if x == len(self.weights)-1:
                    print '============================================='
                    print 'h%i->output weights * h%i = output ==> activation output' % (x, x)
                    print self.weights[x]
                    print '*'
                    print temp_af_sum
                    print '='
                    print node_sum
                    print '==>'
                    print af_sum
                    print '============================================='
                else:
                    print '============================================='
                    print 'h%i->h%i weights * h%i = h%i ==> activation h%i' % (x, x+1, x, x+1, x+1)
                    print self.weights[x]
                    print '*'
                    print temp_af_sum
                    print '='
                    print node_sum
                    print '==>'
                    print af_sum
                    print '============================================='

        if verbose:
            print 'Output: ', af_sum
        return af_sum

    def backPropagation(self, outputData, expected):

        # store error
        self.error.append(float(expected-outputData))

        # start off with backprop for output layer
        delta = np.multiply(expected-outputData, self.dAF(self.nodeValues[-1]))
        weight_change = np.transpose(np.dot(self.activationValues[-2], delta)) * self.LR # last activation value is output, but we want last hidden layer

        # change the weight
        self.weights[-1] += weight_change

        # loop through other hidden layers and update weights
        for x in reversed(xrange(0,len(self.weights)-1)):
            # print np.dot(delta, self.weights[x+1]), self.dAF(self.nodeValues[x+1])
            delta = np.multiply(np.dot(delta, self.weights[x+1]), np.transpose(self.dAF(self.nodeValues[x+1])))
            weight_change = np.transpose(np.dot(self.nodeValues[x], delta)) * self.LR
            self.weights[x] += weight_change


    # trainDataSet is a inputs x N np.array where N is the number of trials
    # expectedResults is a N x 1 np.array which is the expcted results of the trials
    def train(self, trainDataSet, expectedResults):

        # will need to loop through the trainData
        for x in xrange(trainDataSet.shape[1]):
        
            # extract the example
            example = trainDataSet[:, x:x+1]

            # run feedforward
            result = self.feedForward(example)

            # run backprop
            self.backPropagation(result, expectedResults[0, x])

    # in test, there is no changing of weights. results are returned that show
    # the predictions from the neural net in the structure of expctedResults from 
    # the train method. 
    def test(self, testDataSet):
        
        # initialize results
        results = np.zeros((1, testDataSet.shape[1]))

        # loop through test data and run feedforward to generate results
        for x in xrange(testDataSet.shape[1]):

            # generate result
            results[x] = feedForward(testDataSet[:, x:x+1])

        return results

class Plotter:
    """
    This class was made to plot error from the neural networks. This is a good way to 
    plot multiple errors from a bunch of different networks so you can see the performance
    of multiple nets on your training data. It makes it easy to determine which network
    structure will perform well with your given data.

    All you have to do is first create a plotter object (which you can add custom labels to)
    and then add different networks' data with addData(). Make sure to give the network a 
    unique identifier for the descrip so you know which network corresponds to which graph
    in the legend. Once you're ready, just run the plot method and you should see your graphs!

    This class uses the matplotlib.pyplot class to generate plots. 
    """
    import matplotlib.pyplot as plt

    def __init__(self, title="Network Comparison", xaxis="Iterations", yaxis="Error"):
        self.data = []
        self.title = title
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.legend = []

    def addData(self, errorVals, descrip):
        self.data.append(errorVals)
        self.legend.append(descrip)

    def plot(self):
        for x in xrange(len(self.data)):
            plt.plot(range(1,len(self.data[x])+1), self.data[x])
        plt.legend(self.legend)
        plt.xlabel(self.xaxis)
        plt.ylabel(self.yaxis)
        plt.title(self.title)
        plt.show()

# main function to test neural network code
if __name__ == '__main__':
    network = Network()
    network2 = Network(learningRate=.01)
    network3 = Network(hiddens=[3,3])
    trainData = np.zeros((2,10000))
    expectedData = np.zeros((1,10000))
    for x in xrange(trainData.shape[0]):
        trainData[0,x] = 1
        trainData[1,x] = 0
        expectedData[0,x] = np.array([1])
    network.train(trainData, expectedData)
    network2.train(trainData, expectedData)
    network3.train(trainData, expectedData)

    plotter = Plotter()
    plotter.addData(network.error, 'net1')
    plotter.addData(network2.error, 'net2')
    plotter.addData(network3.error, 'net3')
    plotter.plot()