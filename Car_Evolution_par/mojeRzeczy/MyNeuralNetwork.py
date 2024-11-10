import numpy as np


class MyNeuralNetwork:
    
    def __init__ (self):
        
        
        ### reproduction parameters ###
        #_____________________________#
                                    
        # create new link during simulation
        self.probOfNewConnection = 0.8
        
        # value probOfNewConnection is multiplied by on every iteration
        self.probOfNewConnectionMult = 0.95
        
        
        self.minProbOfNewConnection = 0.01



        # delete a connection during simulation
        self.probOfDelConnection = 0.8
        
        # value probOfDelConnection is multiplied by on every iteration
        self.probOfDelConnectionMult = 0.95
        
        
        self.minProbOfDelConnection = 0.01



        # create new neuron
        self.probOfNewNeuron = 0.8
        
        # value probOfNewNeuron is multiplied by on every iteration
        self.probOfNewNeuronMult = 0.99
        
        self.minProbOfNewNeuron = 0.05

        

        # delete a hidden neuron
        #self.probOfDelNeuron = 0.01
        
        # value probOfDelNeuron is multiplied by on every iteration
        #self.probOfDelNeuronMult = 0.9
        
        
        # max standard deviation of values added to weights when creating offspring - 
        # if reproductionStdDev goes to min it goes back up to max
        self.maxReproductionStdDev = 0.5
        
        # standard deviation of values added to weights when creating offspring - changes every reproduction based on mult
        self.reproductionStdDev = self.maxReproductionStdDev

        # value std dev is multiplied by on every iteration
        self.stdDevMult = 0.92
        
        # min standard deviation of values added to weights when creating offspring
        self.minReproductionStdDev = 0.001

        # value min is multiplied by on every StdDev reset to max 
        self.minReproductionStdDevMul = 0.7
        
        #_____________________________#

        ###    network parameters   ###
        #_____________________________#
                                    
        # non zero weight at start - linking input and output
        self.probOfStartConnection = 0.5
        
        # number of tries for creating new neuron at start
        self.startingMutationMagnitude = 2
        
        # standard deviation of normal distr for weights
        self.startingStdDev = 1
        
        # structure of network
        self.nrOfInputs = 5
        self.nrOfHiddenLayers = 3
        self.nrOfOutputs = 4
        
        # number of layers in network, input layer not included
        self.networkSize = self.nrOfHiddenLayers + 1

        # array of sizes of layers (number of neurons for given layer)
        self.layerSizes = np.zeros(self.nrOfHiddenLayers + 2, dtype = int)

        # array of numbers of weights in each neuron for given layer
        self.layerWeightsSizes = np.zeros(self.nrOfHiddenLayers + 2, dtype = int)
        
        # maximum abs walue of a weight
        self.maxAbsWeightVal = 100
        
        #_____________________________#
        
        # core structure of the network
        self.neuralNetwork = []
        
        # network score
        self.fitness = 0
        
        #_____________________________#

        
        # make list of layer sizes
        self.layerSizes[0] = self.nrOfInputs
        self.layerSizes[self.layerSizes.size - 1] = self.nrOfOutputs
        
        # make a list of number of weights in neurons for every layer
        for i in range(1, self.layerWeightsSizes.size):
            self.layerWeightsSizes[i] += 1
            for j in range(0, i):
                self.layerWeightsSizes[i] += self.layerSizes[j]
                
        # make whole network structure for weights
        for i in range(1, self.layerSizes.size):
            self.neuralNetwork.append(np.zeros((self.layerSizes[i], self.layerWeightsSizes[i])).T)
            
        # check for some new neuron mutations
        for i in range(0, self.startingMutationMagnitude):
            if np.random.random() < self.probOfNewNeuron:
                self.addNeuron(int(np.random.random() * self.nrOfHiddenLayers) + 1)
            
        # get the neurons random weights
        for layer in range(0, self.networkSize):
            for neuron in range(0, self.layerSizes[layer + 1]):
                for weight in range(0, self.layerWeightsSizes[layer + 1]):
                    if np.random.random() < self.probOfStartConnection:
                        self.neuralNetwork[layer][weight][neuron] = np.random.normal(0, self.startingStdDev)

    def activationFunction(self, x):
        x_clamped = np.clip(x, -709, 709)  # Prevent overflow in exp
        return 1 / (1 + np.exp(-x_clamped))
        

    def activationFunction2(self, x):
        return 2 * x

    def printNetwork(self):
        print("fitness = ", self.fitness)
        for i in range(0, self.networkSize):
            print(self.neuralNetwork[i])
            print(" ")
            
    def compute(self, input):
        input.insert(0, 1)
        for i in range(0, self.networkSize - 1):
            output = self.activationFunction(np.dot(input, self.neuralNetwork[i]))
            input.extend(output)
        output = self.activationFunction(np.dot(input, self.neuralNetwork[self.networkSize - 1]))
        return output
            
    def addNeuron(self, layerNr):
        newLayerSizes = self.layerSizes.copy()
        
        # update list of layer sizes
        newLayerSizes[layerNr] += 1
        
        # make a list of number of weights in neurons for every layer
        newLayerWeightsSizes = np.zeros(newLayerSizes.size, dtype = int)
        for i in range(1, newLayerWeightsSizes.size):
            newLayerWeightsSizes[i] += 1
            for j in range(0, i):
                newLayerWeightsSizes[i] += newLayerSizes[j]
        
        # make whole network structure for weights
        newNeuralNetwork = []
        for i in range(1, newLayerSizes.size):
            newNeuralNetwork.append(np.zeros((newLayerSizes[i], newLayerWeightsSizes[i])).T)

        # copy the weights
        for layer in range(0, self.networkSize):
            for neuron in range(0, self.layerSizes[layer + 1]):
                newNeuron = 0
                for weight in range(0, self.layerWeightsSizes[layer + 1]):
                    if weight == self.layerWeightsSizes[layerNr+1]:
                        newNeuron = 1
                    newNeuralNetwork[layer][weight + newNeuron][neuron] = self.neuralNetwork[layer][weight][neuron]
                    
        # copy results
        self.layerSizes = newLayerSizes
        self.layerWeightsSizes = newLayerWeightsSizes
        self.neuralNetwork = newNeuralNetwork
        

    def networkCopy(self):
        newNetwork = []
        for i in range(1, self.networkSize + 1):
            newNetwork.append(np.zeros((self.layerSizes[i], self.layerWeightsSizes[i])).T)
        for layer in range(0, self.networkSize):
            for neuron in range(0, self.layerSizes[layer + 1]):
                for weight in range(0, self.layerWeightsSizes[layer + 1]):
                    newNetwork[layer][weight][neuron] = self.neuralNetwork[layer][weight][neuron]
                    if(newNetwork[layer][weight][neuron] > self.maxAbsWeightVal):
                        newNetwork[layer][weight][neuron] = self.maxAbsWeightVal
                    if(newNetwork[layer][weight][neuron] < -self.maxAbsWeightVal):
                        newNetwork[layer][weight][neuron] = -self.maxAbsWeightVal
        return newNetwork

    def copy(self):
        newNeuralNetwork = MyNeuralNetwork()
        newNeuralNetwork.probOfNewConnection = self.probOfNewConnection
        newNeuralNetwork.probOfDelConnection = self.probOfDelConnection
        newNeuralNetwork.probOfNewNeuron = self.probOfNewNeuron
        #newNeuralNetwork.probOfDelNeuron = self.probOfDelNeuron
        newNeuralNetwork.startingStdDev = self.startingStdDev
        newNeuralNetwork.stdDevMult = self.stdDevMult
        newNeuralNetwork.probOfStartConnection = self.probOfStartConnection
        newNeuralNetwork.nrOfInputs = self.nrOfInputs
        newNeuralNetwork.nrOfHiddenLayers = self.nrOfHiddenLayers
        newNeuralNetwork.nrOfOutputs = self.nrOfOutputs
        newNeuralNetwork.networkSize = self.networkSize
        newNeuralNetwork.layerSizes = self.layerSizes.copy()
        newNeuralNetwork.layerWeightsSizes = self.layerWeightsSizes.copy()
        newNeuralNetwork.neuralNetwork = self.networkCopy()
        return newNeuralNetwork
        
    def reproduce(self):
        # copy the parent
        child = self.copy()
        
        # check new neuron mutation
        if np.random.random() < self.probOfNewNeuron:
            child.addNeuron(int(np.random.random() * child.nrOfHiddenLayers) + 1)
        
        # change the weights, make or delete connections
        for layer in range(0, child.networkSize):
            for neuron in range(0, child.layerSizes[layer + 1]):
                for weight in range(0, child.layerWeightsSizes[layer + 1]):
                    if(child.neuralNetwork[layer][weight][neuron] != 0):
                        if np.random.random() < self.probOfDelConnection:
                            child.neuralNetwork[layer][weight][neuron] = 0
                        else:
                            child.neuralNetwork[layer][weight][neuron] += np.random.normal(0, self.reproductionStdDev)
                    else:
                        if np.random.random() < self.probOfNewConnection:
                            child.neuralNetwork[layer][weight][neuron] = np.random.normal(0, self.reproductionStdDev)
        
        # update new parameters for reproduction
        self.probOfNewConnection = max(self.minProbOfNewConnection, self.probOfNewConnection * self.probOfNewConnectionMult)
        self.probOfDelConnection = max(self.minProbOfDelConnection, self.probOfDelConnection * self.probOfDelConnectionMult)
        self.probOfNewNeuron = max(self.minProbOfNewNeuron, self.probOfNewNeuron * self.probOfNewNeuronMult)
        #self.probOfDelNeuron *= self.probOfDelNeuronMult
        self.reproductionStdDev *= self.stdDevMult
        if self.reproductionStdDev < self.minReproductionStdDev:
            self.reproductionStdDev = self.maxReproductionStdDev
            self.minReproductionStdDev *= self.minReproductionStdDevMul
        
        
        return child
    
    def test():

        newNetwork = MyNeuralNetwork()

        # some print
        print(newNetwork.layerSizes)
        print(newNetwork.layerWeightsSizes)
        newNetwork.printNetwork()
        
        newNetwork.readFromFile("bestNNdataModelHalfContinuous.txt")
        
        print(newNetwork.layerSizes)
        print(newNetwork.layerWeightsSizes)
        newNetwork.printNetwork()

        
            
    def sortKey(self):
        return self.fitness
    
    def saveToFile(self, filePath):
        with open(filePath, 'w') as file:
        # Write some values to the file
            file.write("size = \n")
            for i in range(0, self.layerSizes.size):
                file.write(str(self.layerSizes[i]))
                file.write(" ")
            file.write("\n")
            file.write("fitness = \n")
            file.write(str(self.fitness))
            file.write("\n")
            for layer in range(0, self.networkSize):
                file.write("layer nr. ")
                file.write(str(layer + 1))
                file.write("\n")
                if self.layerSizes[layer + 1] == 0:
                        file.write("no neurons \n")
                else:
                    for weight in range(0, self.layerWeightsSizes[layer + 1]):
                        for neuron in range(0, self.layerSizes[layer + 1]):
                            file.write("{:.8f}".format(self.neuralNetwork[layer][weight][neuron], 10))
                            file.write("  ")
                        file.write("\n")
                
                
    def readFromFile(self, filePath):
        with open(filePath, 'r') as file:
            lines = file.readlines()
            # Split the second line into float values
            sizeValues = [float(val) for val in lines[1].split()]
            for i in range(0, len(sizeValues)):
                self.layerSizes[i] = sizeValues[i]
            
            
            self.layerWeightsSizes = np.zeros(self.nrOfHiddenLayers + 2, dtype = int)
            # make a list of number of weights in neurons for every layer
            for i in range(1, self.layerWeightsSizes.size):
                self.layerWeightsSizes[i] += 1
                for j in range(0, i):
                    self.layerWeightsSizes[i] += self.layerSizes[j]
                    
            # make whole network structure for weights
            self.neuralNetwork = []
            for i in range(1, self.layerSizes.size):
                self.neuralNetwork.append(np.zeros((self.layerSizes[i], self.layerWeightsSizes[i])).T)
             
            iterator = 5
            # get the neurons random weights
            for layer in range(0, self.layerSizes.size - 1):
                if self.layerSizes[layer + 1] > 0:
                    for weight in range(0, self.layerWeightsSizes[layer + 1]):
                        weightValues = [float(val) for val in lines[iterator].split()]
                        for neuron in range(0, self.layerSizes[layer + 1]):
                            self.neuralNetwork[layer][weight][neuron] = weightValues[neuron]
                        iterator += 1
                else:
                    iterator += 1
                iterator += 1
            
                
            
              
    

    



