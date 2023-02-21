import datetime
from copy import deepcopy
from functools import partial

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
from obj_grasp.objects import get_obj_hard_ellipsoid
from optmizers_config import get_cfg_graph
from rule_sets.rule_extention_graph import rule_vocab, torque_dict, node_vocab

from rostok.adapters.golem_adapter import (GraphGrammarAdapter,
                                           GraphGrammarFactory)
from rostok.graph_grammar.graph_utils import plot_graph
from rostok.graph_grammar.make_random_graph import make_random_graph
from rostok.graph_grammar.node import GraphGrammar
from rostok.trajectory_optimizer.control_optimizer import ControlOptimizer
import random

def custom_metric(graph: GraphGrammar):
    existing_variables_num = -len(graph)
    print(existing_variables_num)
    return existing_variables_num

def custom_metriwith_build_mechs(graph: GraphGrammar, optimizer: ControlOptimizer):
    res, unused_list = optimizer.start_optimisation(graph)
    print(res)
    return 42

def custom_mutation(graph: GraphGrammar, **kwargs) -> GraphGrammar:
    num_mut = 10
    for _ in range(num_mut):
        rid = random.choice(range(graph.length))
        random_node = graph.nodes[rid]
        other_random_node = graph.nodes[random.choice(range(len(graph.nodes)))]
        graph.connect_nodes(random_node, other_random_node)
    return graph



adapter_local = GraphGrammarAdapter()

rule_vocab = deepcopy(rule_vocab)
init_population_gr = []
for _ in range(15):
    rand_mech = make_random_graph(7, rule_vocab)
    init_population_gr.append(rand_mech)

initial = adapter_local.adapt(init_population_gr)
cfg = get_cfg_graph(torque_dict)
cfg.get_rgab_object_callback = get_obj_hard_ellipsoid
optic = ControlOptimizer(cfg)

build_wrapperd  = partial(custom_metriwith_build_mechs, optimizer = optic)
objective = Objective({'custom': build_wrapperd})
objective_eval = ObjectiveEvaluate(objective)
timeout = datetime.timedelta(minutes = 30)

terminal_nodes = [i for i in list(node_vocab.node_dict.values()) if  i.is_terminal]
nodes_types = adapter_local.adapt_node_seq(terminal_nodes)

graph_generation_params = GraphGenerationParams(
    adapter=adapter_local,
    rules_for_constraint = DEFAULT_DAG_RULES,
    available_node_types=nodes_types,
    node_factory = GraphGrammarFactory(terminal_nodes)
    )


requirements = GraphRequirements(
    max_arity=6,
    max_depth=15,
    parallelization_mode="single",
    timeout=timeout,
    num_of_generations=12,
    history_dir=None)

optimizer_parameters = GPAlgorithmParameters(
    pop_size=15,
    max_pop_size = 12,

    crossover_prob=0.9, mutation_prob=0.5,
    genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        mutation_types=[
            #MutationTypesEnum.growth,
            MutationTypesEnum.none,
            MutationTypesEnum.simple,
            MutationTypesEnum.growth,
            #MutationTypesEnum.single_edge,
            MutationTypesEnum.single_add,
            custom_mutation,
            #MutationTypesEnum.single_add,
            #MutationTypesEnum.single_add,
        ],
    crossover_types = [CrossoverTypesEnum.one_point],
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
rew = optic.create_reward_function(optimized_mech)
rew(optimized_mech, True)