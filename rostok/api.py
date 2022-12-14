import configparser

import pychrono as chrono
import mcts

from rostok.graph_grammar.node import GraphGrammar, BlockWrapper, ROOT
from rostok.block_builder.transform_srtucture import FrameTransform, rotation
from rostok.trajectory_optimizer.control_optimizer import ConfigRewardFunction, ControlOptimizer
from rostok.criterion.flags_simualtions import FlagMaxTime, FlagSlipout, FlagNotContact
from rostok.block_builder.node_render import (LinkChronoBody,MountChronoBody,ChronoTransform,ChronoRevolveJoint,
                                                FlatChronoBody)

import rostok.graph_generators.graph_environment as env
import rostok.graph_grammar.node_vocabulary as node_vocabulary
import rostok.graph_grammar.rule_vocabulary as rule_vocabulary

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

    def create_environment(self, max_number_rules = None):
        """Create environment of searching grab construction. MCTS optimizing environment state with a view to maximizing the rewarCreating an object generating gripping structures. In the `run_generation` method, MCTS optimizes the action in the environment in order to maximize the reward
        """        
        grap_grammar = GraphGrammar()
        if max_number_rules is not None:
            self.max_numbers_non_terminal_rules = max_number_rules
        self.graph_env = env.GraphVocabularyEnvironment(grap_grammar, self.rule_vocabulary, self.max_numbers_non_terminal_rules)
        self.graph_env.set_control_optimizer(self.control_optimizer)

    def run_generation(self, max_search_iteration = None, visualaize=False):
        """Launches the gripping robot generation algorithm .

        Args:
            visualaize (bool, optional):Visualization flag, if true, enables visualization of generation steps. Defaults to False.

        Returns:
            tuple: Tuple of generating result: generate grab mechanism, control trajectory and reward.
        """
        if max_search_iteration is not None:
            self.search_iteration = max_search_iteration
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
    model.rule_vocabulary, model._node_features = create_extension_rules(
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

def create_extension_rules(width_flats:list[float], length_links:list[float]):
    """Creating standard rules: creating palm with 2/3/4 fingers, adding phalanx to mount and terminating node. The function returns the rule_vocabulary is object of `Rule Vocabulary` class. It can manipulate vocabulary rules. Adding new rules or return (non)terminal rules. See description to `rostock.graph_grammar.rule_vocabulary`
    Current, supporting exactly three different width_flats and length_links.

    Args:
        width_flats (list[float]): List of desired flats(palms) width. It has exactly three flat widths 
        length_links (list[float]): List of desired phalanx lengths. It has exactly three link lengths

    Returns:
        tuple: Tuple have two elements: rule_vocabulary(`RuleVocabulary`) and node_features for correct calculate grab criterion.
    """    
    if len(width_flats) != 3 or len(length_links) != 3:
        raise Exception("Read the description. In the current version, you need to add exactly three list items")
    # %% Bodies for extansions rules
    links = list(map(lambda x: BlockWrapper(LinkChronoBody, length=x),
                    length_links))

    flats = list(map(lambda x: BlockWrapper(FlatChronoBody, width=x), width_flats))

    u1 = BlockWrapper(MountChronoBody, width=0.1, length=0.05)
    u2 = BlockWrapper(MountChronoBody, width=0.2, length=0.1)

    MOVE_TO_RIGHT_SIDE = map(lambda x: FrameTransform([x, 0, 0], [0, 0, 1, 0]), width_flats)
    MOVE_TO_RIGHT_SIDE_PLUS = map(lambda x: FrameTransform([x, 0, +0.3], [0, 0, 1, 0]), width_flats)
    MOVE_TO_RIGHT_SIDE_PLUS_ANGLE = map(lambda x: FrameTransform([x, 0, +0.3], rotation(150)),
                                        width_flats)
    MOVE_TO_RIGHT_SIDE_MINUS = map(lambda x: FrameTransform([x, 0, -0.3], [0, 0, 1, 0]), width_flats)
    MOVE_TO_RIGHT_SIDE_MINUS_ANGLE = map(lambda x: FrameTransform([x, 0, -0.3], rotation(210)),
                                         width_flats)
    MOVE_TO_LEFT_SIDE = map(lambda x: FrameTransform([-x, 0, 0], [1, 0, 0, 0]), width_flats)
    MOVE_TO_LEFT_SIDE_PLUS = map(lambda x: FrameTransform([-x, 0, +0.3], [1, 0, 0, 0]), width_flats)
    MOVE_TO_LEFT_SIDE_PLUS_ANGLE = map(lambda x: FrameTransform([-x, 0, +0.3], rotation(30)), width_flats)
    MOVE_TO_LEFT_SIDE_MINUS = map(lambda x: FrameTransform([-x, 0, -0.3], [1, 0, 0, 0]), width_flats)
    MOVE_TO_LEFT_SIDE_MINUS_ANGLE = map(lambda x: FrameTransform([-x, 0, -0.3], rotation(-30)),
                                        width_flats)

    transform_to_right_mount = list(
        map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_RIGHT_SIDE))
    transform_to_right_mount_plus = list(
        map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_RIGHT_SIDE_PLUS))
    transform_to_right_mount_plus_angle = list(
        map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_RIGHT_SIDE_PLUS_ANGLE))
    transform_to_right_mount_minus = list(
        map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_RIGHT_SIDE_MINUS))
    transform_to_right_mount_minus_angle = list(
        map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_RIGHT_SIDE_MINUS_ANGLE))
    transform_to_left_mount = list(
        map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_LEFT_SIDE))
    transform_to_left_mount_plus = list(
        map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_LEFT_SIDE_PLUS))
    transform_to_left_mount_plus_angle = list(
        map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_LEFT_SIDE_PLUS_ANGLE))
    transform_to_left_mount_minus = list(
        map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_LEFT_SIDE_MINUS))
    transform_to_left_mount_minus_angle = list(
        map(lambda x: BlockWrapper(ChronoTransform, x), MOVE_TO_LEFT_SIDE_MINUS_ANGLE))
    # transform_to_alpha_rotate = BlockWrapper(ChronoTransform, ROTATE_TO_ALPHA)

    # %%
    type_of_input = ChronoRevolveJoint.InputType.TORQUE

    # Joints
    revolve1 = BlockWrapper(ChronoRevolveJoint, ChronoRevolveJoint.Axis.Z, type_of_input)

    # Nodes
    node_vocab = node_vocabulary.NodeVocabulary()
    node_vocab.add_node(ROOT)
    node_vocab.create_node("J")
    node_vocab.create_node("L")
    node_vocab.create_node("F")
    node_vocab.create_node("M")
    node_vocab.create_node("EF")
    node_vocab.create_node("EM")
    node_vocab.create_node("SML")
    node_vocab.create_node("SMR")
    node_vocab.create_node("SMRP")
    node_vocab.create_node("SMRPA")
    node_vocab.create_node("SMLP")
    node_vocab.create_node("SMLPA")
    node_vocab.create_node("SMRM")
    node_vocab.create_node("SMRMA")
    node_vocab.create_node("SMLM")
    node_vocab.create_node("SMLMA")

    node_vocab.create_node(label="J1", is_terminal=True, block_wrapper=revolve1)
    
    link_labels = []
    for idx, link in enumerate(links):
        link_labels.append("L" + str(idx+1))
        node_vocab.create_node(label=link_labels[-1], is_terminal=True, block_wrapper=link)
    
    flat_labels = []
    for idx, flat in enumerate(flats):
        flat_labels.append("F" + str(idx+1))
        node_vocab.create_node(label=flat_labels[-1], is_terminal=True, block_wrapper=flat)
        
    node_vocab.create_node(label="U1", is_terminal=True, block_wrapper=u1)
    node_vocab.create_node(label="U2", is_terminal=True, block_wrapper=u2)

    node_vocab.create_node(label="TR1", is_terminal=True, block_wrapper=transform_to_right_mount[0])
    node_vocab.create_node(label="TR2", is_terminal=True, block_wrapper=transform_to_right_mount[1])
    node_vocab.create_node(label="TR3", is_terminal=True, block_wrapper=transform_to_right_mount[2])
    node_vocab.create_node(label="TRP1",
                           is_terminal=True,
                           block_wrapper=transform_to_right_mount_plus[0])
    node_vocab.create_node(label="TRP2",
                           is_terminal=True,
                           block_wrapper=transform_to_right_mount_plus[1])
    node_vocab.create_node(label="TRP3",
                           is_terminal=True,
                           block_wrapper=transform_to_right_mount_plus[2])
    node_vocab.create_node(label="TRPA1",
                           is_terminal=True,
                           block_wrapper=transform_to_right_mount_plus_angle[0])
    node_vocab.create_node(label="TRPA2",
                           is_terminal=True,
                           block_wrapper=transform_to_right_mount_plus_angle[1])
    node_vocab.create_node(label="TRPA3",
                           is_terminal=True,
                           block_wrapper=transform_to_right_mount_plus_angle[2])
    node_vocab.create_node(label="TRM1",
                           is_terminal=True,
                           block_wrapper=transform_to_right_mount_minus[0])
    node_vocab.create_node(label="TRM2",
                           is_terminal=True,
                           block_wrapper=transform_to_right_mount_minus[1])
    node_vocab.create_node(label="TRM3",
                           is_terminal=True,
                           block_wrapper=transform_to_right_mount_minus[2])
    node_vocab.create_node(label="TRMA1",
                           is_terminal=True,
                           block_wrapper=transform_to_right_mount_minus_angle[0])
    node_vocab.create_node(label="TRMA2",
                           is_terminal=True,
                           block_wrapper=transform_to_right_mount_minus_angle[1])
    node_vocab.create_node(label="TRMA3",
                           is_terminal=True,
                           block_wrapper=transform_to_right_mount_minus_angle[2])

    node_vocab.create_node(label="TL1", is_terminal=True, block_wrapper=transform_to_left_mount[0])
    node_vocab.create_node(label="TL2", is_terminal=True, block_wrapper=transform_to_left_mount[1])
    node_vocab.create_node(label="TL3", is_terminal=True, block_wrapper=transform_to_left_mount[2])
    node_vocab.create_node(label="TLP1",
                           is_terminal=True,
                           block_wrapper=transform_to_left_mount_plus[0])
    node_vocab.create_node(label="TLP2",
                           is_terminal=True,
                           block_wrapper=transform_to_left_mount_plus[1])
    node_vocab.create_node(label="TLP3",
                           is_terminal=True,
                           block_wrapper=transform_to_left_mount_plus[2])
    node_vocab.create_node(label="TLPA1",
                           is_terminal=True,
                           block_wrapper=transform_to_left_mount_plus_angle[0])
    node_vocab.create_node(label="TLPA2",
                           is_terminal=True,
                           block_wrapper=transform_to_left_mount_plus_angle[1])
    node_vocab.create_node(label="TLPA3",
                           is_terminal=True,
                           block_wrapper=transform_to_left_mount_plus_angle[2])
    node_vocab.create_node(label="TLM1",
                           is_terminal=True,
                           block_wrapper=transform_to_left_mount_minus[0])
    node_vocab.create_node(label="TLM2",
                           is_terminal=True,
                           block_wrapper=transform_to_left_mount_minus[1])
    node_vocab.create_node(label="TLM3",
                           is_terminal=True,
                           block_wrapper=transform_to_left_mount_minus[2])
    node_vocab.create_node(label="TLMA1",
                           is_terminal=True,
                           block_wrapper=transform_to_left_mount_minus_angle[0])
    node_vocab.create_node(label="TLMA2",
                           is_terminal=True,
                           block_wrapper=transform_to_left_mount_minus_angle[1])
    node_vocab.create_node(label="TLMA3",
                           is_terminal=True,
                           block_wrapper=transform_to_left_mount_minus_angle[2])

    # Defines rules
    rule_vocab = rule_vocabulary.RuleVocabulary(node_vocab)

    rule_vocab.create_rule("InitMechanism_2", ["ROOT"], ["F", "SML", "SMR", "EM", "EM"], 0, 0,
                           [(0, 1), (0, 2), (1, 3), (2, 4)])
    rule_vocab.create_rule("InitMechanism_3_R", ["ROOT"],
                           ["F", "SML", "SMRP", "SMRM", "EM", "EM", "EM"], 0, 0, [(0, 1), (0, 2),
                                                                                  (0, 3), (1, 4),
                                                                                  (2, 5), (3, 6)])
    rule_vocab.create_rule("InitMechanism_3_R_A", ["ROOT"],
                           ["F", "SML", "SMRPA", "SMRMA", "EM", "EM", "EM"], 0, 0, [(0, 1), (0, 2),
                                                                                    (0, 3), (1, 4),
                                                                                    (2, 5), (3, 6)])
    rule_vocab.create_rule("InitMechanism_3_L", ["ROOT"],
                           ["F", "SMLP", "SMLM", "SMR", "EM", "EM", "EM"], 0, 0, [(0, 1), (0, 2),
                                                                                  (0, 3), (1, 4),
                                                                                  (2, 5), (3, 6)])
    rule_vocab.create_rule("InitMechanism_3_L_A", ["ROOT"],
                           ["F", "SMLPA", "SMLMA", "SMR", "EM", "EM", "EM"], 0, 0, [(0, 1), (0, 2),
                                                                                    (0, 3), (1, 4),
                                                                                    (2, 5), (3, 6)])
    rule_vocab.create_rule("FingerUpper", ["EM"], ["J", "L", "EM"], 0, 2, [(0, 1), (1, 2)])

    rule_vocab.create_rule("TerminalFlat1", ["F"], ["F1"], 0, 0)
    rule_vocab.create_rule("TerminalFlat2", ["F"], ["F2"], 0, 0)
    rule_vocab.create_rule("TerminalFlat3", ["F"], ["F3"], 0, 0)

    rule_vocab.create_rule("TerminalL1", ["L"], ["L1"], 0, 0)
    rule_vocab.create_rule("TerminalL2", ["L"], ["L2"], 0, 0)
    rule_vocab.create_rule("TerminalL3", ["L"], ["L3"], 0, 0)

    rule_vocab.create_rule("TerminalTransformRight1", ["SMR"], ["TR1"], 0, 0)
    rule_vocab.create_rule("TerminalTransformRight2", ["SMR"], ["TR2"], 0, 0)
    rule_vocab.create_rule("TerminalTransformRight3", ["SMR"], ["TR3"], 0, 0)

    rule_vocab.create_rule("TerminalTransformRightPlus1", ["SMRP"], ["TRP1"], 0, 0)
    rule_vocab.create_rule("TerminalTransformRightPlus2", ["SMRP"], ["TRP2"], 0, 0)
    rule_vocab.create_rule("TerminalTransformRightPlus3", ["SMRP"], ["TRP3"], 0, 0)

    rule_vocab.create_rule("TerminalTransformRightPlusAngle1", ["SMRPA"], ["TRPA1"], 0, 0)
    rule_vocab.create_rule("TerminalTransformRightPlusAngle2", ["SMRPA"], ["TRPA2"], 0, 0)
    rule_vocab.create_rule("TerminalTransformRightPlusAngle3", ["SMRPA"], ["TRPA3"], 0, 0)

    rule_vocab.create_rule("TerminalTransformRightMinus1", ["SMRM"], ["TRM1"], 0, 0)
    rule_vocab.create_rule("TerminalTransformRightMinus2", ["SMRM"], ["TRM2"], 0, 0)
    rule_vocab.create_rule("TerminalTransformRightMinus3", ["SMRM"], ["TRM3"], 0, 0)

    rule_vocab.create_rule("TerminalTransformRightMinusAngle1", ["SMRMA"], ["TRMA1"], 0, 0)
    rule_vocab.create_rule("TerminalTransformRightMinusAngle2", ["SMRMA"], ["TRMA2"], 0, 0)
    rule_vocab.create_rule("TerminalTransformRightMinusAngle3", ["SMRMA"], ["TRMA3"], 0, 0)

    rule_vocab.create_rule("TerminalTransformLeft1", ["SML"], ["TL1"], 0, 0)
    rule_vocab.create_rule("TerminalTransformLeft2", ["SML"], ["TL2"], 0, 0)
    rule_vocab.create_rule("TerminalTransformLeft3", ["SML"], ["TL3"], 0, 0)

    rule_vocab.create_rule("TerminalTransformLeftPlus1", ["SMLP"], ["TLP1"], 0, 0)
    rule_vocab.create_rule("TerminalTransformLeftPlus2", ["SMLP"], ["TLP2"], 0, 0)
    rule_vocab.create_rule("TerminalTransformLeftPlus3", ["SMLP"], ["TLP3"], 0, 0)

    rule_vocab.create_rule("TerminalTransformLeftPlusAngle1", ["SMLPA"], ["TLPA1"], 0, 0)
    rule_vocab.create_rule("TerminalTransformLeftPlusAngle2", ["SMLPA"], ["TLPA2"], 0, 0)
    rule_vocab.create_rule("TerminalTransformLeftPlusAngle3", ["SMLPA"], ["TLPA3"], 0, 0)

    rule_vocab.create_rule("TerminalTransformLeftMinus1", ["SMLM"], ["TLM1"], 0, 0)
    rule_vocab.create_rule("TerminalTransformLeftMinus2", ["SMLM"], ["TLM2"], 0, 0)
    rule_vocab.create_rule("TerminalTransformLeftMinus3", ["SMLM"], ["TLM3"], 0, 0)

    rule_vocab.create_rule("TerminalTransformLeftMinusAngle1", ["SMLMA"], ["TLMA1"], 0, 0)
    rule_vocab.create_rule("TerminalTransformLeftMinusAngle2", ["SMLMA"], ["TLMA2"], 0, 0)
    rule_vocab.create_rule("TerminalTransformLeftMinusAngle3", ["SMLMA"], ["TLMA3"], 0, 0)

    rule_vocab.create_rule("TerminalEndLimb1", ["EM"], ["U1"], 0, 0)
    rule_vocab.create_rule("TerminalEndLimb2", ["EM"], ["U2"], 0, 0)
    rule_vocab.create_rule("TerminalJoint", ["J"], ["J1"], 0, 0)

    list_J = node_vocab.get_list_of_nodes(["J1"])
    list_RM = node_vocab.get_list_of_nodes([
        "TR1", "TR2", "TR3", "TRP1", "TRP2", "TRP3", "TRM1", "TRM2", "TRM3", "TRPA1", "TRPA2",
        "TRPA3", "TRMA1", "TRMA2", "TRMA3"
    ])
    list_LM = node_vocab.get_list_of_nodes([
        "TL1", "TL2", "TL3", "TLP1", "TLP2", "TLP3", "TLM1", "TLM2", "TLM3", "TLPA1", "TLPA2",
        "TLPA3", "TLMA1", "TLMA2", "TLMA3"
    ])
    list_B = node_vocab.get_list_of_nodes(["L1", "L2", "L3", "F1", "F2", "F3", "U1", "U2"])
    # Required for criteria calc
    node_features = [list_B, list_J, list_LM, list_RM]
    return rule_vocab, node_features