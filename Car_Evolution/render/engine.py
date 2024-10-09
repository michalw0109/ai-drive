# ------------------ IMPORTS ------------------
import pygame
from ai.car_ai import CarAI
from render.car import Car
from render.colors import Color
from mojeRzeczy.Evolution import Evolution
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import cv2

# ------------------ CLASSES ------------------
class Engine:

    WIDTH = 1500
    HEIGHT = 760
    FPS = 0
    
    USE_TRACK_IMAGE = True
    
    DEFAULT_FONT = "comicsansms"
    
    BRUSH_SIZE = 50
    
    

    def __init__(self, MAX_SIMULATIONS, FPS, DATA_MODEL, POPULATION_SIZE, MIN_DURATION, MAX_DURATION, READ_FROM_FILE):
        
        Engine.FPS = FPS
        
        # initializing evolution engine
        self.myEvoEngine = Evolution(POPULATION_SIZE)
        if READ_FROM_FILE:
            match DATA_MODEL:
                case 0:
                    sufix = "dataModelDiscrete"
                case 1:
                    sufix = "dataModelHalfDiscrete"
                case 2:
                    sufix = "dataModelContinuous"
                case 3:
                    sufix = "dataModelHalfContinuous"
            fullPath = "C:/Users/micha/source/python/Car_Evolution/evolutionOutput/bestNN" + sufix + ".txt"
            self.myEvoEngine.createPopulationFromFile(fullPath)
        else:
            self.myEvoEngine.createPopulation()
        
        # graph
        self.fig, self.ax = plt.subplots(1,1)

        self.dataModel = DATA_MODEL
        self.minDuration = MIN_DURATION
        self.maxSimulations = MAX_SIMULATIONS
        
        self.durationStep : float = 1.0 * (MAX_DURATION - MIN_DURATION) / MAX_SIMULATIONS
        
        
        self.title = "Wieczorek Frycz Evolution"
        pygame.display.set_caption(self.title)
        self.screen = pygame.display.set_mode((Engine.WIDTH, Engine.HEIGHT))
        self.screen.fill(Color.WHITE)  # Fill screen with white

        self.is_running = False
        self.clock = pygame.time.Clock()
        
        self.is_drawing_track = True
        self.is_placing_start_point = False
        self.ai_can_start = False
        self.instructions = [
            "Left click to draw a black line, right click to draw a white line. Mouse wheel to adjust brush size. Once you are done drawing, press SPACE to go to the next step.",
            "Use the directional arrows to rotate. Click to place. CTRL + Z to go back to drawing. Once you've placed the start point, the AI will automatically start running.",
        ]
        self.instruction_index = 0
        self.tmp_screen = None
        self.track = None

        self.car = Car([0, 0],[self.WIDTH, self.HEIGHT])
        self.decided_car_pos = None
        
    def handle_drawing_track(self):
        #Handles the drawing of the track
        if(pygame.mouse.get_pressed()[0]):
            pygame.draw.circle(self.screen, Color.BLACK, pygame.mouse.get_pos(), Engine.BRUSH_SIZE)
        elif pygame.mouse.get_pressed()[2]:
            pygame.draw.circle(self.screen, Color.WHITE, pygame.mouse.get_pos(), Engine.BRUSH_SIZE)
            
    def draw_instructions(self):
        #Draws the instructions on the title's screen
        pygame.display.set_caption(self.title + " - " + self.instructions[self.instruction_index])
        
    def handle_placing_start_point(self):
        #Handles the placing of the start point
        # Car sprite follows mouse until left click
        if not pygame.mouse.get_pressed()[0]:
            self.car.position[0] = pygame.mouse.get_pos()[0] - Car.CAR_SIZE_X / 2
            self.car.position[1] = pygame.mouse.get_pos()[1] - Car.CAR_SIZE_Y / 2
            self.screen.blit(self.car.sprite, (self.car.position[0], self.car.position[1]))
 
            pygame.draw.circle(self.screen, Color.RED, (self.WIDTH / 2,self.HEIGHT / 2), 2)
            
        # Once left click is pressed, the car is placed and the AI can start
        else:
            self.ai_can_start = True
            self.is_placing_start_point = False
            self.track = self.screen.copy()
            self.screen.blit(self.car.sprite, (self.car.center[0], self.car.center[1]))
            
            # !!! set car position
            self.decided_car_pos = [1100, 100]

            
            
        
    def run(self) -> None:
        self.running = True
        while self.running:
            
            # Events handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                
                # Handle space bar
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        if self.instruction_index == 0:
                            self.is_drawing_track = False
                            self.is_placing_start_point = True
                            self.instruction_index += 1
                            
                            if self.USE_TRACK_IMAGE:
                                
                                #!!! set default track
                                self.my_track = pygame.image.load("assets/my_track2.png")        
                                self.my_track = pygame.transform.scale(self.my_track, (Engine.WIDTH, Engine.HEIGHT))
                                self.screen.blit(self.my_track, (0,0))

                            
                            
                            self.tmp_screen = self.screen.copy()
                            
                # Handle CTRL + Z when placing start point
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        if self.is_placing_start_point:
                            self.is_placing_start_point = False
                            self.instruction_index -= 1
                            self.is_drawing_track = True

                # Handle scroll brush size
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:
                        # Ensure doesn't go above 100
                        if Engine.BRUSH_SIZE < 100:
                            Engine.BRUSH_SIZE += 1
                    elif event.button == 5:
                        # Ensure doesn't go below 1
                        if Engine.BRUSH_SIZE > 10:
                            Engine.BRUSH_SIZE -= 1
     
            # Draw instructions
            self.draw_instructions()
            
            # Drawing track
            if self.is_drawing_track:
                self.handle_drawing_track()
                
            if self.is_placing_start_point:
                self.handle_placing_start_point()
            

            
            # AI
            if self.ai_can_start:
                self.ai_can_start = False
                self.runMyEvoEngine()
            # Update
            self.update()
            

    def runMyEvoEngine(self):
        
        self.running = False
        counter = 1
        duration : float = self.minDuration
        while(counter < self.maxSimulations):
            
            # Initialize CarAI
            car_ai = CarAI(self.myEvoEngine, self.decided_car_pos,[self.WIDTH, self.HEIGHT], self.dataModel)
            
            # Start timer
            timer = 0

            self.is_running = True
            while self.is_running:
                # Events handler
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()

                # Compute the next generation
                car_ai.compute(self.track)

                # Break if all cars are dead
                if car_ai.remaining_cars == 0:
                    break

            
                if timer > duration * 60:
                    break
                    

                # Draw the track and cars which are still alive
                self.screen.blit(self.track, (0, 0))
                for car in car_ai.cars:
                    car.draw(self.screen)
                    
               
                if timer % 10 == 0:
                    # Refresh and show informations
                    t = "Generation: " + str(car_ai.TOTAL_GENERATIONS)
                    t2 = "Still Alive: " + str(car_ai.remaining_cars)
                    t3 = "Time Left: " + str(round((duration * 60 - timer) / 60, 2)) + "s"
                    t4 = "Best Fitness: " + str(round(car_ai.best_fitness))
                    pygame.display.set_caption(self.title + " - " + t + " - " + t2 + " - " + t3 + " - " + t4)



                # Update the screen
                self.update()
                #self.clock.tick(Engine.FPS)
                timer += 1
            print("generation nr. ", counter)
            print("duration ", duration)
            counter += 1
            duration += self.durationStep
            self.myEvoEngine.nextGeneration(self.dataModel)
            self.graph(self.dataModel)
    
        
        
    def graph(self, mode):
        tick_spacing = int(len(self.myEvoEngine.generationsList) / 10 + 1)

        match mode:
            case 0:
                sufix = "DataModelDiscrete"
            case 1:
                sufix = "DataModelHalfDiscrete"
            case 2:
                sufix = "DataModelContinuous"
            case 3:
                sufix = "DataModelHalfContinuous"


        self.ax.plot(self.myEvoEngine.generationsList, self.myEvoEngine.bestFitnessList)
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        fullPath = "C:/Users/micha/source/python/Car_Evolution/evolutionOutput/bestFitnessGraph" + sufix + ".png"
        plt.savefig(fullPath)
        img = cv2.imread(fullPath, cv2.IMREAD_ANYCOLOR)
        cv2.imshow("Best fitness score", img)
        
        fullDataPath = "C:/Users/micha/source/python/Car_Evolution/evolutionOutput/data" + sufix + ".txt"
        with open(fullDataPath, 'w') as file:
        # Write some values to the file
            for i in range(0, len(self.myEvoEngine.bestFitnessList)):
                file.write(str(self.myEvoEngine.generationsList[i]))
                file.write(" ")
                file.write(str(self.myEvoEngine.bestFitnessList[i]))
                file.write("\n")

        
        


    def update(self):
        pygame.display.update()
        if self.is_placing_start_point:
            self.screen.blit(self.tmp_screen, (0, 0))
            
    def saveSim(self):
        return 0
    