from copy import deepcopy
from typing import Tuple
from rostok.graph_grammar.node import GraphGrammar, Rule
from rostok.graph_grammar.rule_vocabulary import RuleVocabulary
 


def create_graph_from_seq(seq: list[Rule]) -> GraphGrammar:
    """Applies the rules from the list in direct order 

    Parameters
    ----------
    seq : list[Rule] list of rules

    Returns
    -------
    GraphGrammar
        graph
    """
    G = GraphGrammar()
    for i in seq:
        G.apply_rule(i)
    return G


def number_of_non_terminal_rules(seq: list[Rule]) -> int:
    """Calculate number of nonterminal rules
    by using is_terminal method from class graph_grammar.Rule

    Parameters
    ----------
    seq : list[Rule]
         list of rules

    Returns
    -------
    int
        number of nonterminal rules in list
    """
    is_terminal = lambda rule: not rule.is_terminal
    return len(list(filter(is_terminal, seq)))


def _ruleset_explorer(set_uniq_graphs: set[GraphGrammar],
                      limit_non_terminal: int,
                      rule_vocab: RuleVocabulary,
                      mutable_counter: list[int],
                      current_number_non_terminal: int = 0,
                      seq_rule: list[Rule] = []):

    current_graph = create_graph_from_seq(seq_rule)

    if (current_number_non_terminal >= limit_non_terminal):
        rule_names = rule_vocab.get_list_of_applicable_terminal_rules(current_graph)
        
    else:
        rule_names = rule_vocab.get_list_of_applicable_rules(current_graph)
    
    if len(rule_names) == 0:
            set_uniq_graphs.add(current_graph)
            mutable_counter[0] += 1

    for rule_name in rule_names:
        rule = rule_vocab.get_rule(rule_name)
        seq_rule_one = deepcopy(seq_rule)
        seq_rule_one.append(rule)
        number_nt_rules = number_of_non_terminal_rules(seq_rule_one)
        _ruleset_explorer(set_uniq_graphs, limit_non_terminal, rule_vocab, mutable_counter,
                          number_nt_rules, seq_rule_one)

    return current_graph


def ruleset_explorer(limit_non_terminal: int, rule_vocab: RuleVocabulary) -> Tuple[set[GraphGrammar], int]:
    """Recursive iterate over all posible graph in rule_vocab with limitation on non-terminal rules.
    Counts all non-uniq graphs

    Parameters
    ----------
    limit_non_terminal : int
        
    rule_vocab : RuleVocabulary

    Parameters
    ----------
    limit_non_terminal : int
        
    rule_vocab : RuleVocabulary
        

    Returns
    -------
    Tuple[set[GraphGrammar], int]
        first is set of uniq graphs
        last is counter of uniq graphs
    """
    set_uniq_graphs: set[GraphGrammar] = set()
    current_number_non_terminal: int = 0
    seq_rule: list[Rule] = []
    mutable_counter = [0]
 
    _ruleset_explorer(set_uniq_graphs, limit_non_terminal, rule_vocab, mutable_counter, current_number_non_terminal,
                      seq_rule)

    return set_uniq_graphs, mutable_counter[0]


