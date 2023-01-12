from test_ruleset import (get_terminal_graph_three_finger,
                          get_terminal_graph_two_finger,
                          get_terminal_graph_two_finger_mix, rule_vocab)

from rostok.graph_grammar import make_random_graph


def test_graph_equal():
    graph_eq_1 = get_terminal_graph_two_finger_mix()
    graph_eq_2 = get_terminal_graph_two_finger()
    graph3 = get_terminal_graph_three_finger()
    assert graph_eq_1 != graph3
    assert graph_eq_1 == graph_eq_2


def test_create_graph():
    """
        Test for graph grammar and rule
    """
    graph = make_random_graph.make_random_graph(5, rule_vocab)