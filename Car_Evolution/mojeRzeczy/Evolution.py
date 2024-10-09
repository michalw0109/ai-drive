from mojeRzeczy import MyNeuralNetwork as mnn

class Evolution:
    
    def __init__(self, POPULATIONSIZE):
        
        self.populationSize = POPULATIONSIZE
        
        # what fraction of population survives, eg. 0.4 survives, makes another 0.4 with reproduction, 0.2 is new 
        self.survivalRate = 0.4
        
        self.generationsList = [0]

        self.bestFitnessList = [0.0]

        self.population = []
        
    def createPopulation(self):
        for _ in range(0, self.populationSize):
            self.population.append(mnn.MyNeuralNetwork())
            
    def createPopulationFromFile(self, filePath):
        for _ in range(0, self.populationSize):
            self.population.append(mnn.MyNeuralNetwork())
        self.population[0].readFromFile(filePath)
            
    def printPopulation(self):
        for i in range(0, self.populationSize):
            print(" ")
            print("network nr. ", i + 1)
            print(" ")
            self.population[i].printNetwork()
            
    def nextGeneration(self, mode):
        # sort the population
        self.population.sort(key=mnn.MyNeuralNetwork.sortKey, reverse=True)
        
        match mode:
            case 0:
                sufix = "dataModelDiscrete"
            case 1:
                sufix = "dataModelHalfDiscrete"
            case 2:
                sufix = "dataModelContinuous"
            case 3:
                sufix = "dataModelHalfContinuous"
                
                
        fullPath = "C:/Users/micha/source/python/Car_Evolution/evolutionOutput/bestNN" + sufix + ".txt"
        self.population[0].saveToFile(fullPath)
        
        best = self.population[0].fitness
        print("best fitness = "+str(best))
        self.bestFitnessList.extend([best])
        self.generationsList.extend([len(self.generationsList)])
        
        survivorsSize = int(self.populationSize * self.survivalRate)
        
        # reproduce the best
        for i in range (0, survivorsSize):
            self.population[i + survivorsSize] = self.population[i].reproduce()
            
        # fill in the rest with new
        for i in range (0, self.populationSize - 2 * survivorsSize):
            self.population[i + 2 * survivorsSize] = mnn.MyNeuralNetwork()
            
        # reset the fitness
        for i in range (0, self.populationSize):
            self.population[i].fitness = 0
            
    def test(self):
        newPopulation = Evolution()
        newPopulation.createPopulation()
        newPopulation.nextGeneration()
        newPopulation.printPopulation()
        
        
        
        
        

        