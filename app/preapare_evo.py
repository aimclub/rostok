from rostok.graph_grammar.graphgrammar_explorer import random_search_mechs_n_branch

from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.mutation import add_mut, del_mut
from rostok.library.rule_sets import ruleset_old_style_graph
import pickle

rule_vocab, _ = ruleset_old_style_graph.create_rules()
terminal_nodes = rule_vocab.node_vocab.terminal_node_dict.values()


def create_balance_population(rule_vocabul, max_tries=10000):

    two_fingers = random_search_mechs_n_branch(rule_vocabul,
                                       category_size=8,
                                       numbers_of_rules=[6, 8],
                                       desired_branch=2,
                                       max_tries=max_tries)
    three_fingers = random_search_mechs_n_branch(rule_vocabul,
                                         category_size=8,
                                         numbers_of_rules=[8, 10],
                                         desired_branch=3,
                                         max_tries=max_tries)
    four_fingers = random_search_mechs_n_branch(rule_vocabul,
                                        category_size=8,
                                        numbers_of_rules=[10, 11],
                                        desired_branch=4,
                                        max_tries=max_tries)

    return two_fingers + three_fingers + four_fingers


def custom_mutation_add(graph: GraphGrammar, **kwargs) -> GraphGrammar:
    graph_mut = add_mut(graph, terminal_nodes, (1,0,1))
    return graph_mut


def custom_mutation_del(graph: GraphGrammar, **kwargs) -> GraphGrammar:
    graph_mut = del_mut(graph, terminal_nodes, (1,0,1))
    return graph_mut


def load_init_population() -> list[GraphGrammar]:
    try:
        with open('init_population_cache.pickle', 'rb') as f:
            data_new = pickle.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "init_population_cache.pickle. Please generate cache population by using preapare_evo.py"
        ) from exc

    return data_new


if __name__ == "__main__":
    init_population = create_balance_population(rule_vocab)
    with open('init_population_cache.pickle', 'wb') as f:
        pickle.dump(init_population, f)
