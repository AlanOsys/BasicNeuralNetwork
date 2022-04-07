import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weigths = 2 * np.random.random((4,1))-1

    def sigmoid (self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self,x):
        return x * ( 1-x)

    def train(self,training_inputs,training_outputs,training_iterations):
        global arr
        global arr2
        global arr3
        global arr4
        global arr5
        arr = np.zeros(10000)
        arr2 = np.zeros(10000)
        arr3 = np.zeros(10000)
        arr4 = np.zeros(10000)
        arr5 = np.zeros(10000)
        
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error*self.sigmoid_derivative(output))
            self.synaptic_weigths += adjustments
            arr[iteration] = self.synaptic_weigths[0]
            arr2[iteration] = self.synaptic_weigths[1]
            arr3[iteration] = self.synaptic_weigths[2]
            arr4[iteration] = self.synaptic_weigths[3]


    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weigths))
        return output


if __name__ == '__main__':
    
    neural_network = NeuralNetwork()
    training_inputs = np.array([
    [0,0,1,1],
    [1,1,1,0],
    [1,0,1,0],
    [0,1,1,0],
    [0,0,0,0]
    ])
    training_outputs = np.array([[0,1,1,0,1]]).T
    neural_network.train(training_inputs,training_outputs, 10000)
    print(neural_network.synaptic_weigths)
    rarr = arr.reshape(4,2500)
    
    syn = np.arange(10000)
    rsyn = syn.reshape(4,2500)
    
    plt.scatter(syn,arr, c="purple")
    plt.scatter(syn,arr2, c="orange")
    plt.scatter(syn,arr3, c="blue")
    plt.scatter(syn,arr4, c="red")
    
    plt.show()
