import configparser

import app.rule_extention as rules

config = configparser.ConfigParser()
config.read("config.ini")


def init_algorithm(config):
    
    config_rules = config["Links"].join(config["Flats"])
    rule_vocabul, node_features = rules.init_extension_rules(config_rules)
    
    congif_opti_control = config["OptimizingControl"]
