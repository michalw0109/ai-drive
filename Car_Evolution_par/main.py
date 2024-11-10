# ------------------ IMPORTS ------------------
from render.engine import Engine
import numpy as np


# ------------------ GLOBAL VARIABLES ------------------

MAX_SIMULATIONS = 100
FPS = 3000
# 0 - dataModelDiscrete
# 1 - dataModelHalfDiscrete
# 2 - dataModelContinuous
# 3 - dataModelHalfContinuous
DATA_MODEL = 3
POPULATION_SIZE = 100
MIN_DURATION = 10
MAX_DURATION = 20

READ_FROM_FILE = False

# ------------------ MAIN FUNCTION ------------------

def main() -> None:
    np.random.seed(1)
    window = Engine(MAX_SIMULATIONS, FPS, DATA_MODEL, POPULATION_SIZE, MIN_DURATION, MAX_DURATION, READ_FROM_FILE)
    window.runMyEvoEngine()


# ------------------ MAIN CALL ------------------

if __name__ == "__main__":
    main()
