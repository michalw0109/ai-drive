# ------------------ IMPORTS ------------------
import pygame
import math
from render.colors import Color

# ------------------ GLOBAL VARIABLES  ------------------
CAR_SPRITE_PATH = "assets/car.png"
DEAD_CAR_SPRITE_PATH = "assets/dead_car.png"


# ------------------ CLASSES ------------------
class Action:
    TURN_LEFT = 0
    TURN_RIGHT = 1
    ACCELERATE = 2
    BRAKE = 3

class Car:

    CAR_SIZE_X = 20
    CAR_SIZE_Y = 20

    MINIMUM_SPEED = 1

    ANGLE_INCREMENT = 2

    SPEED_INCREMENT = 1

    DEFAULT_SPEED = 1
    DEFAULT_ANGLE = 0

    COLLISION_SURFACE_COLOR = Color.WHITE

    SENSORS_DRAW_DISTANCE = 1920

    def __init__(self, start_position: list, screen_dim: list):
        
        self.screen_WIDTH = screen_dim[0]
        self.screen_HEIGHT = screen_dim[1]
        

        self.position = start_position.copy()

        self.angle = Car.DEFAULT_ANGLE
        self.speed = Car.DEFAULT_SPEED

        self.center = [
            self.position[0] + Car.CAR_SIZE_X / 2,
            self.position[1] + Car.CAR_SIZE_Y / 2
        ]  # Calculate Center

        self.sensors = []
        self.alive = True
        self.has_been_rendered_as_dead = False
        
        self.driven_distance = 0
        self.speed_penalty = 0

        self.max_angle = 0
        self.laps = 0
        self.current_angle = 0
        
        

    

    def check_collision(self, track) -> bool:
        #Check if the car is colliding with the track (by using a color system)

        track_x = track.get_width()
        track_y = track.get_height()
        for point in self.corners:
            if point[0] < 0 or point[0] >= track_x or point[1] < 0 or point[1] >= track_y:
                self.alive = False
                return True

            elif track.get_at((int(point[0]), int(point[1]))) == Car.COLLISION_SURFACE_COLOR:
                self.alive = False
                return True

        return False

    def refresh_corners_positions(self) -> None:
        #Refresh the corners' current positions of the car (used for collision detection)
        length_x = 0.5 * Car.CAR_SIZE_X
        length_y = 0.5 * Car.CAR_SIZE_Y

        corner1 = math.radians(360 - (self.angle + 30))
        corner2 = math.radians(360 - (self.angle + 150))
        corner3 = math.radians(360 - (self.angle + 210))
        corner4 = math.radians(360 - (self.angle + 330))

        left_top = [
            self.center[0] + math.cos(corner1) * length_x,
            self.center[1] + math.sin(corner1) * length_y
        ]
        right_top = [
            self.center[0] + math.cos(corner2) * length_x,
            self.center[1] + math.sin(corner2) * length_y
        ]
        left_bottom = [
            self.center[0] + math.cos(corner3) * length_x,
            self.center[1] + math.sin(corner3) * length_y
        ]
        right_bottom = [
            self.center[0] + math.cos(corner4) * length_x,
            self.center[1] + math.sin(corner4) * length_y
        ]

        self.corners = [left_top, right_top, left_bottom, right_bottom]

    def check_sensor(self, degree: int, track) -> None:
        #Check the distance between the center of the car and the collision surface to create the sensors

        # Convert degree to radians because math.cos and math.sin use radians
        radians = math.radians(360 - (self.angle + degree))
        cos = math.cos(radians)
        sin = math.sin(radians)
        length = 1

        x, y = int(self.center[0]), int(self.center[1])

        track_x = track.get_width()
        track_y = track.get_height()

        # While the collision surface is not reached, increment the length of the sensor
        while x < track_x and y < track_y and x > 0 and y > 0 and track.get_at((x, y)) != Car.COLLISION_SURFACE_COLOR:
            x = int(self.center[0] + cos * length)
            y = int(self.center[1] + sin * length)

            # If the max length of a sensor is reached, break the loop
            if length > Car.SENSORS_DRAW_DISTANCE:
                break

            length += 1

        # Distance calculation between the center of the car and the collision surface
        distance = int(math.hypot(x - self.center[0], y - self.center[1]))
        self.sensors.append([(x, y), distance])

    def update_center(self) -> None:
        #Update the center of the car after a rotation (when it turns left or right)

        # Calculate New Center
        self.center = [
            int(self.position[0]) + Car.CAR_SIZE_X / 2,
            int(self.position[1]) + Car.CAR_SIZE_Y / 2
        ]

    def update_sprite(self, track) -> None:
        #Update the sprite of the car and its new informations (position, center, sensors, etc.)

        # Update the sprite
        self.update_center()

        # Radians, cos, sin
        radians = math.radians(360 - self.angle)
        cos = math.cos(radians)
        sin = math.sin(radians)

        # Move car to new position
        self.position[0] += cos * self.speed
        self.position[1] += sin * self.speed
        
        # Update the driven distance with the speed
        self.driven_distance += self.speed
        
        #count the number of laps
        if self.position[0] - self.screen_WIDTH / 2 != 0:
            temp = math.atan((self.position[1] - self.screen_HEIGHT / 2)/(self.position[0] - self.screen_WIDTH / 2)) + math.pi/2
            if (self.current_angle - temp) > 3:
                self.laps += 1
            if (temp - self.current_angle) > 3:
                self.laps -= 1
            self.current_angle = temp
            if self.current_angle + self.laps * math.pi > self.max_angle:
                self.max_angle = self.current_angle + self.laps * math.pi
        
      
        
        # Calculate Corners
        self.refresh_corners_positions()

        # Check collisions
        self.check_collision(track)

        # Clear radars and rewrite them (-50, -25, 0, 25, 50)
        self.sensors.clear()
        for sensor_angle in range(-50, 50 + 1, 25):
            self.check_sensor(sensor_angle, track)

    def get_data(self) -> list[int]:
        #Get the data of the car's sensors

        # Get distances to border
        distances = [int(sensor[1]) for sensor in self.sensors]

        # Ensure list has five elements (to correspond to)
        distances += [0] * (5 - len(distances))

        return distances

    def get_reward(self) -> float:

        # Reward for distance driven
        final_reward = 100 * self.max_angle
  
        return final_reward

    
    def accelerate(self, mult) -> None:
        #Accelerate the car
        self.speed += Car.SPEED_INCREMENT * mult

    def brake(self, mult) -> None:
        #Brake the car
        self.speed -= Car.SPEED_INCREMENT * mult
        if self.speed < Car.MINIMUM_SPEED:  # We don't want to go backwards nor going too slow
            self.speed = Car.MINIMUM_SPEED
        
    def turn_left(self, mult) -> None:
        #Turn the car to the left
        self.angle += Car.ANGLE_INCREMENT * mult
        
    def turn_right(self, mult) -> None:
        #Turn the car to the right
        self.angle -= Car.ANGLE_INCREMENT * mult