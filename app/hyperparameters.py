MAX_NUMBER_RULES = 13

BASE_ITERATION_LIMIT = 30
BASE_ITERATION_LIMIT_GRAPH = 200
BASE_ITERATION_LIMIT_TENDON = 100

ITERATION_REDUCTION_TIME = 0.7

TIME_CRITERION_WEIGHT = 3
FORCE_CRITERION_WEIGHT = 1
OBJECT_COG_CRITERION_WEIGHT = 1
INSTANT_FORCE_CRITERION_WEIGHT = 1
INSTANT_CONTACTING_LINK_CRITERION_WEIGHT = 1
GRASP_TIME_CRITERION_WEIGHT = 1
FINAL_POSITION_CRITERION_WEIGHT = 5
REFERENCE_DISTANCE = 0.3

CONTROL_OPTIMIZATION_BOUNDS = (0.0001, 0.002)
CONTROL_OPTIMIZATION_BOUNDS_TENDON = (10, 20)
CONTROL_OPTIMIZATION_ITERATION = 8
CONTROL_OPTIMIZATION_ITERATION_TENDON = 20
TENDON_CONST = -5

TIME_STEP_SIMULATION = 0.0001
GRASP_TIME = 1
FORCE_TEST_TIME = 1
TIME_SIMULATION = 2

FLAG_TIME_NO_CONTACT = 1
FLAG_TIME_SLIPOUT = 1
FLAG_FLYING_APART = 10