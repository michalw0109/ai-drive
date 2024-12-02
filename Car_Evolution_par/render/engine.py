# ------------------ IMPORTS ------------------
import pygame
from ai.car_ai import CarAI
from render.car import Car
from render.colors import Color
from mojeRzeczy.Evolution import Evolution
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import cv2
from datetime import datetime
import os
import multiprocessing
from PIL import Image





def parrarel_generation(start_index, length, _myEvoEngine, _decided_car_pos, _dims: list, _dataModel, _duration):
    
    try:
        # Kod generacji
        # Initialize CarAI
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
        
        pygame.display.set_mode((1,1))

        car_ai = CarAI(_myEvoEngine, _decided_car_pos, _dims, _dataModel, start_index, length)
        
        temp_track = pygame.image.load("assets/my_track2.png")        
        temp_track = pygame.transform.scale(temp_track, (_dims[0], _dims[1]))


        # Start timer
        timer = 0

        _is_running = True
        while _is_running:
            

            # Compute the next generation
            car_ai.compute(temp_track)

            # Break if all cars are dead
            if car_ai.remaining_cars == 0:
                break

            if timer > _duration * 60:
                break
     
            timer += 1
            
        
        return start_index, [individual.fitness for individual in _myEvoEngine.population[start_index:start_index+length]]

    finally:
        pygame.quit()  # Zamknięcie pygame dla każdego procesu
    

# ------------------ CLASSES ------------------
class Engine:

    WIDTH = 1500
    HEIGHT = 760
    FPS = 0
    PATH_TO_FOLDER = "/home/deweloper/Pulpit/ai_drive/1/Car_Evolution_par/"
    
    USE_TRACK_IMAGE = True
    
    DEFAULT_FONT = "comicsansms"
    
    BRUSH_SIZE = 50
    
    NR_OF_THREADS = 4
    
    

    def __init__(self, MAX_SIMULATIONS, FPS, DATA_MODEL, POPULATION_SIZE, MIN_DURATION, MAX_DURATION, READ_FROM_FILE):
        


        
        
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


        Engine.FPS = FPS
        self.time = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        # initializing evolution engine
        self.myEvoEngine = Evolution(POPULATION_SIZE, self.time)
        match DATA_MODEL:
            case 0:
                sufix = "dataModelDiscrete"
            case 1:
                sufix = "dataModelHalfDiscrete"
            case 2:
                sufix = "dataModelContinuous"
            case 3:
                sufix = "dataModelHalfContinuous"
                
        os.makedirs(self.PATH_TO_FOLDER + "evolutionOutput/" + sufix + "_" + self.time, exist_ok=True)

        if READ_FROM_FILE:
            
            fullPath = self.PATH_TO_FOLDER + "evolutionOutput/bestNN.txt"
            self.myEvoEngine.createPopulationFromFile(fullPath)
        else:
            self.myEvoEngine.createPopulation()
        
        # graph
        self.fig, self.ax = plt.subplots(1,1)

        self.dataModel = DATA_MODEL
        self.minDuration = MIN_DURATION
        self.maxSimulations = MAX_SIMULATIONS
        
        self.durationStep : float = 1.0 * (MAX_DURATION - MIN_DURATION) / MAX_SIMULATIONS
        
        
        self.title = "Car Evolution"
        pygame.display.set_caption(self.title)
        self.screen = pygame.display.set_mode((1, 1))
        self.screen.fill(Color.WHITE)  # Fill screen with white

        self.is_running = False
        self.clock = pygame.time.Clock()
        

        self.ai_can_start = True

        self.tmp_screen = None
        self.track = None

        self.car = Car([0, 0],[self.WIDTH, self.HEIGHT])
        self.decided_car_pos = [1100, 100]
        
    
    
   
   
            
     
    

    

    def runMyEvoEngine(self):
        
        counter = 0
        duration : float = self.minDuration
        while(counter < self.maxSimulations):
            counter += 1

            # Prepare data for processing: split into chunks
            chunks = [( i * int(self.myEvoEngine.populationSize / Engine.NR_OF_THREADS),
                        int(self.myEvoEngine.populationSize / Engine.NR_OF_THREADS),
                        self.myEvoEngine,
                        self.decided_car_pos,
                        [self.WIDTH, self.HEIGHT],
                        self.dataModel,
                        duration) 
                    for i in range(Engine.NR_OF_THREADS)]
          

            # Create a pool of worker processes
            with multiprocessing.Pool(processes=Engine.NR_OF_THREADS) as pool:
                # Map the worker function to the chunks of data
                results = pool.starmap(parrarel_generation, chunks)

            # Initialize an empty array to hold the results

            # Collect the results and restore original order
            for starting_index, result in results:
                for i, value in enumerate(result):
                    self.myEvoEngine.population[starting_index + i].fitness = value

    
            
            
            print("generation nr. ", counter)
            print("duration ", format(duration, ".3f"))
            duration += self.durationStep
            self.myEvoEngine.nextGeneration(self.dataModel)
            self.graph(self.dataModel)
    
        
        
    def graph(self, mode):
        tick_spacing = int(len(self.myEvoEngine.generationsList) / 10 + 1)

        match mode:
            case 0:
                sufix = "dataModelDiscrete"
            case 1:
                sufix = "dataModelHalfDiscrete"
            case 2:
                sufix = "dataModelContinuous"
            case 3:
                sufix = "dataModelHalfContinuous"


        self.ax.plot(self.myEvoEngine.generationsList, self.myEvoEngine.bestFitnessList)
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        fullPath = self.PATH_TO_FOLDER + "evolutionOutput/"+ sufix + "_" + self.time + "/" + "bestFitnessGraph.png"
        
        plt.savefig(fullPath)
        #img = cv2.imread(fullPath, cv2.IMREAD_ANYCOLOR)
        #cv2.imshow("Best fitness score", img)
        
        fullDataPath = self.PATH_TO_FOLDER + "evolutionOutput/"+ sufix +"_"+ self.time+ "/" + "graphData.txt"
        with open(fullDataPath, 'w') as file:
        # Write some values to the file
            for i in range(0, len(self.myEvoEngine.bestFitnessList)):
                file.write(str(self.myEvoEngine.generationsList[i]))
                file.write(" ")
                file.write(str(self.myEvoEngine.bestFitnessList[i]))
                file.write("\n")


    
            
    def saveSim(self):
        return 0
    