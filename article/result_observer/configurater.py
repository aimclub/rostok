
from pathlib import Path

from rostok.trajectory_optimizer.control_optimizer import ControlOptimizer
from rostok.utils.pickle_save import load_saveable
import optmizers_config as optmizers_config

# %% Graph-OldRule-Zateyniki
def configure_graph_old_zateynik():
    from article.obj_grasp.objects import get_obj_hard_mesh_zateynik
    from article.rule_sets import ruleset_old_style_graph_nonails

    rule_vocabul, torque_dict = ruleset_old_style_graph_nonails.create_rules()
    cfg = optmizers_config.get_cfg_graph(torque_dict)
    cfg.get_rgab_object_callback = get_obj_hard_mesh_zateynik
    control_optimizer = ControlOptimizer(cfg)
    report = load_saveable(Path(r"results\Reports_23y_02m_27d_12H_42M_graph_oldrule_zateyniki\MCTS_data_windows.pickle"))
    return report, control_optimizer, rule_vocabul

# %% Graph-OldRule-Mikki
def configure_graph_old_mikki():
    from article.obj_grasp.objects import get_obj_hard_mesh_mikki
    from article.rule_sets import ruleset_old_style_graph_nonails

    rule_vocabul, torque_dict = ruleset_old_style_graph_nonails.create_rules()
    cfg = optmizers_config.get_cfg_graph(torque_dict)
    cfg.get_rgab_object_callback = get_obj_hard_mesh_mikki
    control_optimizer = ControlOptimizer(cfg)
    report = load_saveable(Path(r"results\Reports_23y_02m_27d_13H_46M_graph_oldrule_mikki\MCTS_data_windows.pickle"))
    return report, control_optimizer, rule_vocabul

# %% Graph-OldRule-Pyramid
def configure_graph_old_pyramida():
    from article.obj_grasp.objects import get_obj_hard_mesh_piramida
    from article.rule_sets import ruleset_old_style_graph_nonails

    rule_vocabul, torque_dict = ruleset_old_style_graph_nonails.create_rules()
    cfg = optmizers_config.get_cfg_graph(torque_dict)
    cfg.get_rgab_object_callback = get_obj_hard_mesh_piramida
    control_optimizer = ControlOptimizer(cfg)
    report = load_saveable(Path(r"results\Reports_23y_02m_27d_11H_49M_graph_oldrule_pyramida\MCTS_data_windows.pickle"))
    return report, control_optimizer, rule_vocabul

# %% Graph-NewRule-Zateyniki    
def configure_graph_new_zateynik():
    from article.obj_grasp.objects import get_obj_hard_mesh_zateynik
    from article.rule_sets import ruleset_new_style_graph_nonails


    rule_vocabul, torque_dict = ruleset_new_style_graph_nonails.create_rules()

    cfg = optmizers_config.get_cfg_graph(torque_dict)
    cfg.get_rgab_object_callback = get_obj_hard_mesh_zateynik    
    report = load_saveable(Path(r"results\Reports_23y_02m_27d_14H_05M_graph_newrule_zateyniki\MCTS_data_windows.pickle"))
    control_optimizer = ControlOptimizer(cfg)    
    return report, control_optimizer, rule_vocabul

# %% Graph-NewRule-Mikki
def configure_graph_new_mikki():
    from article.obj_grasp.objects import get_obj_hard_mesh_mikki
    from article.rule_sets import ruleset_new_style_graph_nonails

    rule_vocabul, torque_dict = ruleset_new_style_graph_nonails.create_rules()
    cfg = optmizers_config.get_cfg_graph(torque_dict)
    cfg.get_rgab_object_callback = get_obj_hard_mesh_mikki
    control_optimizer = ControlOptimizer(cfg)
    report = load_saveable(Path(r"results\Reports_23y_02m_28d_23H_14M_graph_new_rule_mikki\MCTS_data_windows.pickle"))
    return report, control_optimizer, rule_vocabul


# %% Grapg-Newrule-Pyramid    
def configure_graph_new_piramida():
    from article.obj_grasp.objects import get_obj_hard_mesh_piramida
    from article.rule_sets import ruleset_new_style_graph_nonails

    rule_vocabul, torque_dict = ruleset_new_style_graph_nonails.create_rules()
    cfg = optmizers_config.get_cfg_graph(torque_dict)
    cfg.get_rgab_object_callback = get_obj_hard_mesh_piramida
    control_optimizer = ControlOptimizer(cfg)
    report = load_saveable(Path(r"results\Reports_23y_02m_27d_18H_46M_graph_newrule_pyramida\MCTS_data_windows.pickle"))
    return report, control_optimizer, rule_vocabul

def configure_graph_new_ellipsoid():
    from article.obj_grasp.objects import get_obj_hard_large_ellipsoid
    from article.rule_sets import ruleset_new_style_graph_nonails

    rule_vocabul, torque_dict = ruleset_new_style_graph_nonails.create_rules()
    cfg = optmizers_config.get_cfg_graph(torque_dict)
    cfg.get_rgab_object_callback = get_obj_hard_large_ellipsoid
    control_optimizer = ControlOptimizer(cfg)
    report = load_saveable(Path(r"results\Reports_23y_02m_28d_19H_28M_graph_newrule_ellipsoid\MCTS_data_windows.pickle"))
    return report, control_optimizer, rule_vocabul

def configure_graph_new_puck():
    from article.obj_grasp.objects import get_obj_easy_cylinder
    from article.rule_sets import ruleset_new_style_graph_nonails

    rule_vocabul, torque_dict = ruleset_new_style_graph_nonails.create_rules()
    cfg = optmizers_config.get_cfg_graph(torque_dict)
    cfg.get_rgab_object_callback = get_obj_easy_cylinder
    control_optimizer = ControlOptimizer(cfg)
    report = load_saveable(Path(r"results\Reports_23y_02m_28d_19H_53M_graph_newtule_puck\MCTS_data_windows.pickle"))
    return report, control_optimizer, rule_vocabul


def configure_graph_old_ellipsoid():
    from article.obj_grasp.objects import get_obj_hard_large_ellipsoid
    from article.rule_sets import ruleset_old_style_graph_nonails

    rule_vocabul, torque_dict = ruleset_old_style_graph_nonails.create_rules()
    cfg = optmizers_config.get_cfg_graph(torque_dict)
    cfg.get_rgab_object_callback = get_obj_hard_large_ellipsoid
    control_optimizer = ControlOptimizer(cfg)
    report = load_saveable(Path(r"results\Reports_23y_02m_28d_20H_06M_graph_oldrule_ellipsoid\MCTS_data_windows.pickle"))
    return report, control_optimizer, rule_vocabul

def configure_graph_old_puck():
    from article.obj_grasp.objects import get_obj_easy_cylinder
    from article.rule_sets import ruleset_old_style_graph_nonails

    rule_vocabul, torque_dict = ruleset_old_style_graph_nonails.create_rules()
    cfg = optmizers_config.get_cfg_graph(torque_dict)
    cfg.get_rgab_object_callback = get_obj_easy_cylinder
    control_optimizer = ControlOptimizer(cfg)
    report = load_saveable(Path(r"results\Reports_23y_02m_28d_21H_04M_graph_oldrule_puck\MCTS_data_windows.pickle"))
    return report, control_optimizer, rule_vocabul


def configure_graph_new_cylinder():
    from article.obj_grasp.objects import get_obj_easy_cylinder
    from article.rule_sets import ruleset_new_style_graph_nonails

    rule_vocabul, torque_dict = ruleset_new_style_graph_nonails.create_rules()
    cfg = optmizers_config.get_cfg_graph(torque_dict)
    cfg.get_rgab_object_callback = get_obj_easy_cylinder
    control_optimizer = ControlOptimizer(cfg)
    report = load_saveable(Path(r"results\Reports_23y_03m_01d_05H_01M_graph_newrule_cylinder\MCTS_data_windows.pickle"))
    return report, control_optimizer, rule_vocabul

def configure_graph_old_cylinder():
    from article.obj_grasp.objects import get_obj_easy_cylinder
    from article.rule_sets import ruleset_old_style_graph_nonails

    rule_vocabul, torque_dict = ruleset_old_style_graph_nonails.create_rules()
    cfg = optmizers_config.get_cfg_graph(torque_dict)
    cfg.get_rgab_object_callback = get_obj_easy_cylinder
    control_optimizer = ControlOptimizer(cfg)
    report = load_saveable(Path(r"results\Reports_23y_03m_01d_04H_36M_graph_oldrule_cylinder\MCTS_data_windows.pickle"))
    return report, control_optimizer, rule_vocabul


def configure_graph_old_brusok():
    from article.obj_grasp.objects import get_obj_easy_long_tilt_box
    from article.rule_sets import ruleset_old_style_graph_nonails

    rule_vocabul, torque_dict = ruleset_old_style_graph_nonails.create_rules()
    cfg = optmizers_config.get_cfg_graph(torque_dict)
    cfg.get_rgab_object_callback = get_obj_easy_long_tilt_box
    control_optimizer = ControlOptimizer(cfg)
    report = load_saveable(Path(r"results\Reports_23y_03m_01d_06H_16M_graph_odlrule_brusok\MCTS_data_windows.pickle"))
    return report, control_optimizer, rule_vocabul


def configure_graph_new_brusok():
    from article.obj_grasp.objects import get_obj_easy_long_tilt_box
    from article.rule_sets import ruleset_new_style_graph_nonails

    rule_vocabul, torque_dict = ruleset_new_style_graph_nonails.create_rules()
    cfg = optmizers_config.get_cfg_graph(torque_dict)
    cfg.get_rgab_object_callback = get_obj_easy_long_tilt_box
    control_optimizer = ControlOptimizer(cfg)
    report = load_saveable(Path(r"results\Reports_23y_03m_01d_05H_14M_graph_newrule_brusok\MCTS_data_windows.pickle"))
    return report, control_optimizer, rule_vocabul


def configure_old_puck():
    from article.obj_grasp.objects import get_obj_easy_puck
    from article.rule_sets import ruleset_old_style_nonails

    rule_vocabul = ruleset_old_style_nonails.create_rules()
    cfg = optmizers_config.get_cfg_standart()
    cfg.get_rgab_object_callback = get_obj_easy_puck
    control_optimizer = ControlOptimizer(cfg)
    report = load_saveable(Path(r"results\Reports_23y_03m_01d_17H_07M_oldrule_puck\MCTS_data_windows.pickle"))
    return report, control_optimizer, rule_vocabul