# ------------------ IMPORTS ------------------
from render.engine import Engine
import numpy as np
import warnings

# ------------------ GLOBAL VARIABLES ------------------

MAX_SIMULATIONS = 500
FPS = 3000
# 0 - dataModelDiscrete
# 1 - dataModelHalfDiscrete
# 2 - dataModelContinuous
# 3 - dataModelHalfContinuous
DATA_MODEL = 3
POPULATION_SIZE = 100
MIN_DURATION = 200
MAX_DURATION = 50

READ_FROM_FILE = False

# ------------------ MAIN FUNCTION ------------------

def main() -> None:
    np.random.seed(1)
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)  
    #warnings.filterwarnings("ignore", category=RuntimeWarning)
    window = Engine(MAX_SIMULATIONS, FPS, DATA_MODEL, POPULATION_SIZE, MIN_DURATION, MAX_DURATION, READ_FROM_FILE)
    window.runMyEvoEngine()


# ------------------ MAIN CALL ------------------

if __name__ == "__main__":
    main()