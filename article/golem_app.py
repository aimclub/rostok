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
from obj_grasp.objects import get_obj_hard_ellipsoid, get_object_to_grasp_sphere
from optmizers_config import get_cfg_graph
from rule_sets.ruleset_old_style_graph import create_rules

from rostok.adapters.golem_adapter import (GraphGrammarAdapter,
                                           GraphGrammarFactory)
from rostok.graph_grammar.graph_utils import plot_graph
from rostok.graph_grammar.make_random_graph import make_random_graph
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import ControlOptimizer
import random
from mutation_logik import add_mut, del_mut
from golem.core.optimisers.genetic.operators import crossover


rule_vocab, torque_dict = create_rules()
node_vocab = rule_vocab.node_vocab
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

rule_vocab = deepcopy(rule_vocab)
init_population_gr = []
for _ in range(120):
    rand_mech = make_random_graph(12, rule_vocab)
    init_population_gr.append(rand_mech)

initial = adapter_local.adapt(init_population_gr)
cfg = get_cfg_graph(torque_dict)
cfg.get_rgab_object_callback = get_object_to_grasp_sphere
optic = ControlOptimizer(cfg)

build_wrapperd  = partial(custom_metriwith_build_mechs, optimizer = optic)
objective = Objective({'custom': build_wrapperd})
objective_eval = ObjectiveEvaluate(objective)
timeout = datetime.timedelta(hours = 8)

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
    max_arity=4,
    max_depth=17,
    parallelization_mode="single",
    timeout=timeout,
    num_of_generations=40,
    early_stopping_iterations = 20,
    history_dir=None)

optimizer_parameters = GPAlgorithmParameters(
    pop_size = 120,
    max_pop_size = 120,
    

    crossover_prob=0.5, mutation_prob=0.7,
    genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free,
        mutation_types=[
            #MutationTypesEnum.growth,
            MutationTypesEnum.none,
            #MutationTypesEnum.simple,
            #MutationTypesEnum.growth,
            #MutationTypesEnum.single_edge,
            #MutationTypesEnum.single_add,
            custom_mutation_add,
            custom_mutation_del,
            #MutationTypesEnum.single_add,
            #MutationTypesEnum.single_add,
        ],
    crossover_types = [CrossoverTypesEnum.gg_subtree],
    regularization_type = RegularizationTypesEnum.none,
    mutation_strength = MutationStrengthEnum.mean
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
plot_graph(optimized_mech)
name = str(int(time.time()))
with open(name, 'wb') as handle:
    pickle.dump(optimizer.history, handle)
rew = optic.create_reward_function(optimized_mech)
rew(optimized_mech, True)