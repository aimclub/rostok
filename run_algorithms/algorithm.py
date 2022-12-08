import configparser

import pychrono as chrono
import mcts

from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import ConfigRewardFunction, ControlOptimizer
from rostok.criterion.flags_simualtions import FlagMaxTime, FlagSlipout, FlagNotContact
import rostok.graph_generators.graph_environment as env

import app.rule_extention as rules
from app.control_optimisation import create_grab_criterion_fun, create_traj_fun, get_object_to_grasp

# config = configparser.ConfigParser()
# config.read("config.ini")

class OpenChainGen:
    def __init__(self, config):
        self.config = config
        
    def init_control_optimizer(self, config_optimze, node_features):
        bound = (float(config_optimze["low_bound"]), float(config_optimze["up_bound"]))
        cfg = ConfigRewardFunction()
        cfg.bound = bound
        cfg.iters = int(config_optimze["iteration"])
        cfg.sim_config = {"Set_G_acc": chrono.ChVectorD(0, 0, 0)}
        cfg.time_step = float(config_optimze["time_step"])
        cfg.time_sim = float(config_optimze["time_sim"])
        cfg.flags = [
            FlagMaxTime(cfg.time_sim),
            FlagNotContact(cfg.time_sim / 3),
            FlagSlipout(cfg.time_sim / 3, 0.25)
        ]

        criterion_callback = create_grab_criterion_fun(node_features, float(config_optimze["gait"]),
                                                    float(config_optimze["weight"]))
        traj_generator_fun = create_traj_fun(cfg.time_sim, cfg.time_step)

        cfg.criterion_callback = criterion_callback
        cfg.get_rgab_object_callback = get_object_to_grasp
        cfg.params_to_timesiries_callback = traj_generator_fun

        control_optimizer = ControlOptimizer(cfg)
        return control_optimizer


    def init_search_algrorithm(self, config_search, rule_vocabul, control_optimizer):
        searcher = mcts.mcts(iterationLimit=int(config_search["iteration"]))
        G = GraphGrammar()
        max_rules = int(config_search["max_rules"])
        graph_env = env.GraphVocabularyEnvironment(G, rule_vocabul, max_rules)
        graph_env.set_control_optimizer(control_optimizer)
        return graph_env


    def init_algorithm(self, config):

        config_rules = config["Links"].join(config["Flats"])
        rule_vocabul, node_features = rules.init_extension_rules(config_rules)

        congif_opti_control = config["OptimizingControl"]
        control = init_control_optimizer(congif_opti_control, node_features)

        config_search = config["MCTS"]
        env = init_search_algrorithm(config_search, rule_vocabulary, control_optimizer)
