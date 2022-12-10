import configparser

import pychrono as chrono
import mcts

from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import ConfigRewardFunction, ControlOptimizer
from rostok.criterion.flags_simualtions import FlagMaxTime, FlagSlipout, FlagNotContact
import rostok.graph_generators.graph_environment as env

import app.rule_extention as rules
from app.control_optimisation import create_grab_criterion_fun, create_traj_fun, get_object_to_grasp


class OpenChainGen:
    """The main class manipulate settings and running generation open chain grab mechanism
    
    There are two ways to start robot generation. The first method uses the configuration file `rostock/config.ini` with the function `api.create_generator_by_config'.
    It returns the class of the generation algorithm specified by the configuration. It remains to call the 'run_generation` method, which launches the search capture of the robot. In the second method, you configure the class arguments yourself.
    Further, there are minimalistic descriptive arguments of the class.
    
        
    Args:
        control_optimizer (ControlOptimizer): Object manipulate control optimizing. Defaults to None.
        graph_env (GraphEnvironment): Object manipulate MCTS environment. Defaults to None.
        rule_vocabulary (RuleVocabulary): Vocabulary of graph grammar rules. Defaults to None.
        stop_simulation_flags (StopSimulationFlags): Flags for stopping simulation by some condition. Defaults to None.
        search_iteration (int):  The maximum number of non-terminal rules that can be applied. Defaults to 0.
        max_numbers_non_terminal_rules (int): The maximum number of non-terminal rules that can be applied. Defaults to 0.
    """    
    def __init__(self):
        self.control_optimizer = None
        self.graph_env = None
        self.rule_vocabulary = None
        self._node_features = None
        self.stop_simulation_flags = None
        self.search_iteration = 0
        self.max_numbers_non_terminal_rules = 0

    def create_control_optimizer(self, bound, iterations, time_step, time_sim, gait):
        """Creating a control optimization object based on input data

        Args:
            bound (tuple): The lower and upper limit of the input robot control. The format is (min, max)
            iterations (int): Maximum number of optimizing iteration
            time_step (float): Step width of simulation for optimizing control
            time_sim (float): Define maximum time of simulation for optimizing control
            gait (float): Time value of grasping's gait period
        """        
        WEIGHT = [5, 0, 1, 9]

        cfg = ConfigRewardFunction()
        cfg.bound = bound
        cfg.iters = iterations
        cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
        cfg.time_step = time_step
        cfg.time_sim = time_sim
        cfg.flags = self.stop_simulation_flags

        criterion_callback = create_grab_criterion_fun(self._node_features, gait, WEIGHT)
        traj_generator_fun = create_traj_fun(cfg.time_sim, cfg.time_step)

        cfg.criterion_callback = criterion_callback
        cfg.get_rgab_object_callback = get_object_to_grasp
        cfg.params_to_timesiries_callback = traj_generator_fun

        self.control_optimizer = ControlOptimizer(cfg)

    def create_environment(self):
        """Create environment of searching grab construction. MCTS optimizing environment state with a view to maximizing the rewarCreating an object generating gripping structures. In the `run_generation` method, MCTS optimizes the action in the environment in order to maximize the reward
        """        
        G = GraphGrammar()
        max_rules = self.max_numbers_non_terminal_rules
        self.graph_env = env.GraphVocabularyEnvironment(G, self.rule_vocabulary, max_rules)
        self.graph_env.set_control_optimizer(self.control_optimizer)

    def run_generation(self, visualaize=False):
        """Launches the gripping robot generation algorithm .

        Args:
            visualaize (bool, optional):Visualization flag, if true, enables visualization of generation steps. Defaults to False.

        Returns:
            tuple: Tuple of generating result: generate grab mechanism, control trajectory and reward.
        """
        iter = 0
        finish = False
        searcher = mcts.mcts(iterationLimit=self.search_iteration)
        while not finish:
            action = searcher.search(initialState=self.graph_env)
            finish, final_graph, opt_trajectory = self.graph_env.step(action, visualaize)
            iter += 1
            print(
                f"number iteration: {iter}, counter actions: {self.graph_env.counter_action}, reward: {self.graph_env.reward}"
            )
        return final_graph, opt_trajectory, self.graph_env.reward


def create_generator_by_config(config_file: str) -> OpenChainGen:
    """Creating a mechanism generation object from a configuration file
    
    After creation, you can change the object for your task or just run a search (`run_generation`).
    Example of a configuration file in the folder `./rosrok/config.ini`

    Args:
        config_file (str): Path to config file by format .ini

    Returns:
        OpenChainGen: The mechanism generation object
    """
    model = OpenChainGen()
    config = configparser.ConfigParser()
    config.read(config_file)

    config_links = config["Links"]
    config_flats = config["Flats"]

    widths_flat = list(map(lambda x: float(x), config_flats["width"].split(",")))
    lengths_link = list(map(lambda x: float(x), config_links["length"].split(",")))
    model.rule_vocabulary, model._node_features = rules.create_extension_rules(
        widths_flat, lengths_link)

    congif_opti_control = config["OptimizingControl"]
    bound = (float(congif_opti_control["low_bound"]), float(congif_opti_control["up_bound"]))
    iteration_opti_control = int(congif_opti_control["iteration"])
    time_sim = float(congif_opti_control["time_sim"])
    time_step = float(congif_opti_control["time_step"])
    gait = float(congif_opti_control["gait"])
    model.stop_simulation_flags = [
        FlagMaxTime(time_sim),
        FlagSlipout(time_sim / 4, 0.5),
        FlagNotContact(time_sim / 4)
    ]
    model.create_control_optimizer(bound, iteration_opti_control,
                                                             time_step, time_sim, gait)

    config_search = config["MCTS"]
    model.search_iteration = int(config_search["iteration"])
    model.max_numbers_non_terminal_rules = int(config_search["max_non_terminal_rules"])
    model.create_environment()

    return model
