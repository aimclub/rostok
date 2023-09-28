import datetime
import pickle
import time
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters, MutationAgentTypeEnum, ElitismTypesEnum, SelectionTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import \
    GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.base_mutations import (MutationStrengthEnum,
                                                                    tree_growth)
from golem.core.optimisers.genetic.operators.regularization import \
    RegularizationTypesEnum
from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams

from rostok.adapters.golem_adapter import (GraphGrammarAdapter, GraphGrammarFactory)

from rostok.graph_grammar.crossovers import subtree_crossover
from preapare_evo import custom_mutation_del_body, load_init_population, terminal_nodes, \
create_balance_population, rule_vocab, custom_mutation_del_j, custom_mutation_del_tr, \
custom_mutation_add_body, custom_mutation_add_j, custom_mutation_add_tr
from evaluate_graph import mock_with_build_mech
from mcts_run_setup import config_with_standard_graph
from rostok.library.obj_grasp.objects import get_object_box, get_object_cylinder
from rostok.library.rule_sets import ruleset_old_style_graph

boxer  = get_object_cylinder(0.55, 0.8,0)
rules, torque_dict = ruleset_old_style_graph.create_rules()
optic = config_with_standard_graph(boxer, torque_dict)


def fun(x):
    res = optic.calculate_reward(x)
    return -res[0]


CACHED_POPULATION = True
TIMEOUT_MINUTES = 60 * 2

# Define objective
name_objective = get_object_box.__name__
objective = Objective({name_objective: fun})
objective_eval = ObjectiveEvaluate(objective)

# Create init population
init_population_gr = []
if CACHED_POPULATION:
    init_population_gr = load_init_population()
else:
    init_population_gr = create_balance_population(rule_vocab)

# Adapt nodes and init population
adapter_local = GraphGrammarAdapter()
adapted_initial_population = adapter_local.adapt(init_population_gr)
adapted_nodes_types = adapter_local.adapt_node_seq(terminal_nodes)

graph_generation_params = GraphGenerationParams(adapter=adapter_local,
                                                rules_for_constraint=DEFAULT_DAG_RULES,
                                                available_node_types=adapted_nodes_types,
                                                node_factory=GraphGrammarFactory(terminal_nodes))

timeout = datetime.timedelta(minutes=TIMEOUT_MINUTES)
requirements = GraphRequirements(
    max_arity=5,
    min_arity=1,
    max_depth=17,
    # parallelization_mode='single',
    n_jobs=4,
    timeout=timeout,
    num_of_generations=60,
    early_stopping_iterations=30,
    history_dir=None,
    keep_n_best=3)

optimizer_parameters = GPAlgorithmParameters(
    pop_size=len(adapted_initial_population),
    max_pop_size=len(adapted_initial_population) + 10,
    crossover_prob=0.5,
    mutation_prob=0.5,
    adaptive_mutation_type = MutationAgentTypeEnum.default, 
    genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free,
    mutation_types=[custom_mutation_del_body, custom_mutation_del_j, custom_mutation_del_tr,  
                    custom_mutation_add_body, custom_mutation_add_j, custom_mutation_add_tr,
                    tree_growth],
    crossover_types=[subtree_crossover],
    regularization_type=RegularizationTypesEnum.none,
    mutation_strength=MutationStrengthEnum.strong,
    elitism_type= ElitismTypesEnum.keep_n_best,
    offspring_rate = 0.35,
    selection_types=(SelectionTypesEnum.tournament,))

optimizer = EvoGraphOptimizer(objective=objective,
                              initial_graphs=adapted_initial_population,
                              requirements=requirements,
                              graph_generation_params=graph_generation_params,
                              graph_optimizer_params=optimizer_parameters)

optimized_graphs = optimizer.optimise(objective_eval)
name = str(int(time.time()))
name2 = str(optimizer.history.final_choices.data[0].fitness)
name2 = name2.replace(".", "_")
name = name + name_objective + name2
with open(name, 'wb') as handle:
    pickle.dump(optimizer.history, handle)
