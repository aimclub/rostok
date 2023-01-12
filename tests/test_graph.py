from test_ruleset import get_terminal_graph_two_finger_mix, \
get_terminal_graph_two_finger, get_terminal_graph_three_finger


def test_graph_equal():
    graph_eq_1 = get_terminal_graph_two_finger_mix()
    graph_eq_2 = get_terminal_graph_two_finger()
    graph3 = get_terminal_graph_three_finger()
    assert graph_eq_1 != graph3
    assert graph_eq_1 == graph_eq_2