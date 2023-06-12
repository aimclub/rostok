import datetime
import pickle
import time
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.inheritance import \
    GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.mutation import (MutationStrengthEnum)
from golem.core.optimisers.genetic.operators.regularization import \
    RegularizationTypesEnum
from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams

from rostok.adapters.golem_adapter import (GraphGrammarAdapter, GraphGrammarFactory)

from rostok.graph_grammar.crossovers import subtree_crossover
from app.golem_case.preapare_evo import load_init_population, terminal_nodes, custom_mutation_add, custom_mutation_del, create_balance_population, rule_vocab
from app.golem_case.evaluate_graph import mock_with_build_mech

CACHED_POPULATION = True
TIMEOUT_MINUTES = 5

# Define objective
name_objective = mock_with_build_mech.__name__
objective = Objective({name_objective: mock_with_build_mech})
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
requirements = GraphRequirements(max_arity=5,
                                 min_arity=1,
                                 max_depth=17,
                                 parallelization_mode="single",
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
    genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free,
    mutation_types=[custom_mutation_add, custom_mutation_del],
    crossover_types=[subtree_crossover],
    regularization_type=RegularizationTypesEnum.none,
    mutation_strength=MutationStrengthEnum.mean)

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
