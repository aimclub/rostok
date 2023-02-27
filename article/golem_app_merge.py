import datetime
from copy import deepcopy
from functools import partial
import pickle
import time
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import \
    CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import \
    GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.mutation import (
    MutationStrengthEnum, MutationTypesEnum)
from golem.core.optimisers.genetic.operators.regularization import \
    RegularizationTypesEnum
from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from obj_grasp.objects import get_obj_hard_mesh_piramida, get_object_to_grasp_sphere, get_obj_hard_large_ellipsoid, get_obj_cyl_pos_parametrize, get_obj_hard_ellipsoid_move
from optmizers_config import get_cfg_graph
from rule_sets.ruleset_old_style_graph import create_rules
from rule_sets.rule_extention_merge import rule_vocab, torque_dict, node_vocab
from rostok.adapters.golem_adapter import (GraphGrammarAdapter,
                                           GraphGrammarFactory)
from rostok.graph_grammar.graph_utils import plot_graph
from rostok.graph_grammar.make_random_graph import make_random_graph
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import ControlOptimizer
import random
from mutation_logik import add_mut, del_mut
from golem.core.optimisers.genetic.operators import crossover
import init_pop_2
from random import choice

def custom_metric(graph: GraphGrammar):
    existing_variables_num = -len(graph)
    print(existing_variables_num)
    return existing_variables_num

def custom_metriwith_build_mechs(graph: GraphGrammar, optimizer: ControlOptimizer):
    res, unused_list = optimizer.start_optimisation(graph)
    #print(res)
    return res

def custom_crossover(graph_first, graph_second, **kwargs):
    return graph_first, graph_second 

adapter_local = GraphGrammarAdapter()

 
init_population_gr = []
for _ in range(20):
    numes = choice([5, 8, 10])
    rand_mech = make_random_graph(numes, rule_vocab)
    init_population_gr.append(rand_mech)

init_population_gr = init_pop_2.get_population_zoo() 
initial = adapter_local.adapt(init_population_gr)
cfg = get_cfg_graph(torque_dict)
cfg.get_rgab_object_callback = get_obj_hard_mesh_piramida
optic = ControlOptimizer(cfg)

build_wrapperd  = partial(custom_metriwith_build_mechs, optimizer = optic)
name_objective = cfg.get_rgab_object_callback.__name__
objective = Objective({name_objective: build_wrapperd})
objective_eval = ObjectiveEvaluate(objective)
timeout = datetime.timedelta(hours = 10)

terminal_nodes = [i for i in list(node_vocab.node_dict.values()) if  i.is_terminal]
nodes_types = adapter_local.adapt_node_seq(terminal_nodes)

def custom_mutation_add(graph: GraphGrammar, **kwargs) -> GraphGrammar:
    try:
        graph_mut = add_mut(graph, terminal_nodes)
    except:
        graph_mut = deepcopy(graph)
  
    return graph_mut
def custom_mutation_del(graph: GraphGrammar, **kwargs) -> GraphGrammar:
    try:
        graph_mut = del_mut(graph, terminal_nodes)
    except:
        graph_mut = deepcopy(graph)
    return graph_mut

graph_generation_params = GraphGenerationParams(
    adapter=adapter_local,
    rules_for_constraint = DEFAULT_DAG_RULES,
    available_node_types=nodes_types,
    node_factory = GraphGrammarFactory(terminal_nodes)
    )


requirements = GraphRequirements(
    max_arity=5,
    min_arity=1,
    max_depth=17,
    early_stopping_timeout = 9000,
    parallelization_mode="single",
    timeout=timeout,
    num_of_generations=80,
    early_stopping_iterations = 60,
    history_dir=None)

optimizer_parameters = GPAlgorithmParameters(
    pop_size = len(initial),
    max_pop_size = len(initial)+10,
    crossover_prob=0.5, mutation_prob=0.8,
    genetic_scheme_type=GeneticSchemeTypesEnum.generational,
        mutation_types=[
            MutationTypesEnum.none,
            custom_mutation_add,
            custom_mutation_del
        ],
    crossover_types = [CrossoverTypesEnum.gg_subtree],
    regularization_type = RegularizationTypesEnum.decremental,
    mutation_strength = MutationStrengthEnum.mean,
    )

optimizer = EvoGraphOptimizer(
        objective=objective,
        initial_graphs=initial,
        requirements=requirements,
        graph_generation_params=graph_generation_params,
        graph_optimizer_params=optimizer_parameters

        )

optimized_graphs = optimizer.optimise(objective_eval)
optimized_mech = adapter_local.restore(optimized_graphs[0])
#plot_graph(optimized_mech)
#optimized_graphs[0].
name = str(int(time.time()))
name2 = str(optimizer.history.final_choices.data[0].fitness)
name2 = name2.replace(".", "_")
name2 = name2.replace("-", "_")
name = name + name_objective + name2
with open(name, 'wb') as handle:
    pickle.dump(optimizer.history, handle)
