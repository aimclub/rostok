import app.app_vocabulary as voca


def test_passing1():
    print("First")
    assert (1, 2, 3) == (1, 2, 3)


def test_stub2():
    print("Second")
    graph1 = voca.get_random_graph(5, voca.app_rule_vocab)
    assert (1, 2, 3) == (1, 2, 3)