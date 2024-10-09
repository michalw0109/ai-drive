# ------------------ IMPORTS ------------------
from render.engine import Engine


# ------------------ GLOBAL VARIABLES ------------------

MAX_SIMULATIONS = 200
FPS = 3000
# 0 - dataModelDiscrete
# 1 - dataModelHalfDiscrete
# 2 - dataModelContinuous
# 3 - dataModelHalfContinuous
DATA_MODEL = 3
POPULATION_SIZE = 20
MIN_DURATION = 5
MAX_DURATION = 30

READ_FROM_FILE = False

# ------------------ MAIN FUNCTION ------------------

def main() -> None:
    window = Engine(MAX_SIMULATIONS, FPS, DATA_MODEL, POPULATION_SIZE, MIN_DURATION, MAX_DURATION, READ_FROM_FILE)
    window.run()


# ------------------ MAIN CALL ------------------

if __name__ == "__main__":
    main()
