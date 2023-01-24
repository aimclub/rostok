import configparser
from pathlib import Path
from typing import Callable, Optional, Any

import pychrono as chrono
import mcts

from rostok.graph_grammar.node import GraphGrammar
import rostok.block_builder.envbody_shapes as env_shape
from rostok.block_builder.node_render import Material, DefaultChronoMaterial
from rostok.block_builder.transform_srtucture import FrameTransform
from rostok.trajectory_optimizer.control_optimizer import ConfigRewardFunction, ControlOptimizer
import rostok.criterion.flags_simualtions as flags
import rostok.graph_generators.graph_environment as env
import rostok.graph_grammar.rule_vocabulary as rule_vocabulary
from rostok.graph_generators.mcts_helper import MCTSGraphEnviromnent, prepare_mcts_state_and_helper, make_mcts_step

import rostok.launch.control.control_optimisation as ctrl_opt
from rostok.launch.open_chain_grasper_set_rules_ver_1 import get_234_fingers_mechanism_rules


class OpenChainGen:
    """The main class manipulate settings and running generation open chain grab mechanism
    
    There are two ways to start robot generation. The first method uses the 
    configuration file `rostock/launch/config.ini` with the function :py:func:`api.create_generator_by_config`.
    It returns the class of the generation algorithm specified by the configuration. It remains to call the :py:method:'OpenChainGen.run_generation` 
    method, which launches the search capture of the robot. In the second method, you configure the class arguments yourself.
    Further, there are minimalistic descriptive arguments of the class.

    Attributes:
        graph_env (MCTSGraphEnviromnent): Object manipulate MCTS environment. Defaults to None.
        rule_vocabulary (RuleVocabulary): Vocabulary of graph grammar rules. Defaults to None.
        stop_simulation_flags (StopSimulationFlags): Flags for stopping simulation by some condition. Defaults to empty list.
        search_iteration (int):  The maximum number of non-terminal rules that can be applied. Defaults to 0.
        max_numbers_non_terminal_rules (int): The maximum number of non-terminal rules that can be applied. Defaults to 0.
    """

    def __init__(self) -> None:
        self.__complete_generation = False
        self._cfg_control_optimizer: ConfigRewardFunction = ConfigRewardFunction()
        self.graph_env: Optional[MCTSGraphEnviromnent] = None
        self.rule_vocabulary: rule_vocabulary.RuleVocabulary = rule_vocabulary.RuleVocabulary()
        self._node_features: list = [[]]
        self._builder_grasp_object: Optional[Callable[[], Any]] = None
        self.stop_simulation_flags: list[flags.FlagStopSimualtions] = list()
        self.search_iteration: int = 0
        self.max_numbers_non_terminal_rules: int = 0
        self.path_to_result: Path = Path("./results")

    @property
    def cfg_control_optimizer(self):
        """ ConfigRewardFunction: The config of control optimizer
        """
        return self._cfg_control_optimizer

    @cfg_control_optimizer.setter
    def control_optimizer(self, value: ControlOptimizer):
        self._control_optimizer = value
        self._cfg_control_optimizer.get_rgab_object_callback = self._builder_grasp_object

    def set_grasp_object(self,
                         shape: env_shape.ShapeTypes = env_shape.Box(),
                         position: FrameTransform = FrameTransform([0, 1.5, 0],
                                                                   [0, -0.048, 0.706, 0.706]),
                         material: Material = DefaultChronoMaterial()):
        """Setter a grasp object 

        Args:
            grasp_object (SimpleBody): Desired object to grasp
        """
        self._builder_grasp_object = ctrl_opt.create_builder_grasp_object(shape, position, material)
        self._cfg_control_optimizer.get_rgab_object_callback = self._builder_grasp_object

    def set_settings_control_optimizer(self, bound, iterations, time_step, time_sim, gait,
                                       criterion_weights):
        """Creating a control optimization object based on input data

        Args:
            bound (tuple): The lower and upper limit of the input robot control. The format is (min, max)
            iterations (int): Maximum number of optimizing iteration
            time_step (float): Step width of simulation for optimizing control
            time_sim (float): Define maximum time of simulation for optimizing control
            gait (float): Time value of grasping's gait period
            weights_criterion (list[float]):
        """
        self._cfg_control_optimizer = ConfigRewardFunction()
        self._cfg_control_optimizer.bound = bound
        self._cfg_control_optimizer.iters = iterations
        self._cfg_control_optimizer.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
        self._cfg_control_optimizer.time_step = time_step
        self._cfg_control_optimizer.time_sim = time_sim
        self._cfg_control_optimizer.flags = self.stop_simulation_flags

        criterion_callback = ctrl_opt.create_grab_criterion_fun(self._node_features, gait,
                                                                criterion_weights)
        traj_generator_fun = ctrl_opt.create_traj_fun(self._cfg_control_optimizer.time_sim,
                                                      self._cfg_control_optimizer.time_step)

        self._cfg_control_optimizer.criterion_callback = criterion_callback
        self._cfg_control_optimizer.get_rgab_object_callback = self._builder_grasp_object
        self._cfg_control_optimizer.params_to_timesiries_callback = traj_generator_fun

    def set_config_control_optimizer(self, config: ConfigRewardFunction):
        self._cfg_control_optimizer = config

    def run_generation(self, max_search_iteration: int = 0, visualaize: bool = False):
        """Launches the gripping robot generation algorithm.

        Args:
            max_search_iteration int: The maximum number of iterations of the Monte Carlo tree search exploration at each step. Defaults to 0.
            visualaize bool: Visualization flag, if true, enables visualization of generation steps. Defaults to False.

        Raises:
            Exception: The grab object is not specified before the algorithm is started

        Returns:
            tuple: Tuple of generating result: generate grab mechanism, control trajectory and reward.
        """
        control_optimizer = ControlOptimizer(self._cfg_control_optimizer)

        if self._builder_grasp_object is None:
            raise Exception("Object to grasp wasn't set")

        if max_search_iteration != 0:
            self.search_iteration = max_search_iteration

        self.graph_env = prepare_mcts_state_and_helper(GraphGrammar(), self.rule_vocabulary,
                                                       control_optimizer,
                                                       self.max_numbers_non_terminal_rules,
                                                       self.path_to_result)
        mcts_helper = self.graph_env.helper

        n_steps = 0
        finish = False
        searcher = mcts.mcts(iterationLimit=self.search_iteration)
        while not finish:
            finish, self.graph_env = make_mcts_step(searcher, self.graph_env, n_steps)
            n_steps += 1
            print(
                f"number iteration: {n_steps}, counter actions: {self.graph_env.counter_action} " +
                f"reward: {mcts_helper.report.get_best_info()[1]}")
        self.__complete_generation = True
        return mcts_helper.report

    def visualize_result(self):
        if not self.__complete_generation:
            raise Exception("Don't have results. Before visualize results you must run algorithm")
        reporter = self.graph_env.helper.report
        reporter.draw_best_graph()
        reporter.plot_means()
        best_graph, reward, best_control = reporter.get_best_info()
        ctrl_opt = ControlOptimizer(self._cfg_control_optimizer)
        func_reward = ctrl_opt.create_reward_function(best_graph, True)
        func_reward(best_control)

    def save_result(self):
        reporter = self.graph_env.helper.report
        reporter.make_time_dependent_path()
        reporter.save()
        reporter.save_all()


def create_generator_by_config(config_file: str) -> OpenChainGen:
    """Creating a mechanism generation object from a configuration file
    
    After creation, need to set the object for your task (:py:method:`OpenChainGen.set_grasp_object`)s and run a search (:py:method:`OpenChainGen.run_generation`).
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

    widths_flat = [float(x) for x in config_flats["width"].split(",")]
    lengths_link = [float(x) for x in config_links["length"].split(",")]
    model.rule_vocabulary, model._node_features = get_234_fingers_mechanism_rules(
        widths_flat, lengths_link)

    congif_opti_control = config["OptimizingControl"]

    low_bound = float(congif_opti_control["low_bound"])
    up_bound = float(congif_opti_control["up_bound"])
    bound = (low_bound, up_bound)

    iteration_opti_control = int(congif_opti_control["iteration"])
    time_step = float(congif_opti_control["time_step"])
    gait = float(congif_opti_control["gait"])
    criterion_weights = [float(x) for x in congif_opti_control["weights"].split(",")]

    flag_config = config["StopFlagSimulation"]

    stop_flags: list[flags.FlagStopSimualtions] = []
    for str_prefix_flag in flag_config["flags"].split(","):
        str_prefix_flag = str_prefix_flag.replace(" ", "")
        if str_prefix_flag == "MaxTime":
            stop_flags.append(flags.FlagMaxTime(float(flag_config["time_sim"])))
            continue
        if str_prefix_flag == "Slipout":
            stop_flags.append(
                flags.FlagSlipout(float(flag_config["time_with_no_contact"]),
                                  float(flag_config["time_slipout_error"])))
            continue
        if str_prefix_flag == "NotContact":
            stop_flags.append(flags.FlagNotContact(float(flag_config["time_with_no_contact"])))
            continue

    model.stop_simulation_flags = stop_flags

    time_sim = float(flag_config["time_sim"])

    model.set_settings_control_optimizer(bound, iteration_opti_control, time_step, time_sim, gait,
                                         criterion_weights)

    config_search = config["MCTS"]
    model.search_iteration = int(config_search["iteration"])
    model.max_numbers_non_terminal_rules = int(config_search["max_non_terminal_rules"])

    return model