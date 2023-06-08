import datetime
from copy import deepcopy
from functools import partial
import pickle
import time
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.inheritance import \
    GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.mutation import (
    MutationStrengthEnum, MutationTypesEnum)
from golem.core.optimisers.genetic.operators.regularization import \
    RegularizationTypesEnum
from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from rostok.library.rule_sets.ruleset_old_style_graph import create_rules
from rostok.adapters.golem_adapter import (GraphGrammarAdapter,
                                           GraphGrammarFactory)
from rostok.graph_grammar.make_random_graph import make_random_graph
from rostok.graph_grammar.node import GraphGrammar
from mutation_logik import add_mut, del_mut
import init_pop2 as init_pop_2
from random import choice
from crossovers_builder import subtree_crossover
from mcts_run_setup import config_with_standard_graph
from rostok.library.obj_grasp.objects import get_object_parametrized_cylinder, get_object_easy_box, get_obj_hard_get_obj_hard_large_ellipsoid
from rostok.library.rule_sets.rule_extention_merge import rule_vocab, torque_dict

def custom_metric(graph: GraphGrammar):
    existing_variables_num = -len(graph)
    print(existing_variables_num)
    return existing_variables_num



adapter_local = GraphGrammarAdapter()

#rule_vocab, torque_dict = create_rules()
node_vocab = rule_vocab.node_vocab 
init_population_gr = []
for _ in range(32):
    numes = choice([8, 10, 11])
    rand_mech = make_random_graph(numes, rule_vocab)
    init_population_gr.append(rand_mech)

init_population_gr = init_pop_2.get_population_zoo() 
initial = adapter_local.adapt(init_population_gr)
obj = get_object_parametrized_cylinder(0.4, 1, 0.7)
reward_graph = config_with_standard_graph(obj, torque_dict)

build_wrapperd  = reward_graph.count_reward

def build_wrapperd_r(graph):
    res, _ = reward_graph.count_reward(graph)
    return -res

name_objective = "get_object_parametrized_cylinder"
objective = Objective({name_objective: build_wrapperd_r})
objective_eval = ObjectiveEvaluate(objective)
timeout = datetime.timedelta(minutes= 45)

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
    crossover_prob=0.2, mutation_prob=0.5,
    genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free,
        mutation_types=[
            custom_mutation_add,
            custom_mutation_del
        ],
    crossover_types = [subtree_crossover],
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