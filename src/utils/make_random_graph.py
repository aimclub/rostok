from engine.node import GraphGrammar
from engine import rule_vocabulary
import numpy as np


def make_random_graph(n_iter: int, rule_vocab: rule_vocabulary.RuleVocabulary, use_nonterminal_only: bool=True):
    """Return random graph made with n_iter rules
    
    At each step applied a random rule from a list of rules appliable for the current grahp. 
    If use_nonterminal_only is True all applied rules are True, if it is Flase, first half of rules will be nonterminal and
    the second half will be random terminal and nonterminal. 
    """
    G = GraphGrammar()
    for _ in range(n_iter//2+1):
        rules = rule_vocab.get_list_of_applicable_nonterminal_rules(G)
        if len(rules)>0:
            rule = rule_vocab.get_rule(rules[np.random.choice(len(rules))])
            G.apply_rule(rule)
        else: break
    
    for _ in range(n_iter-(n_iter//2+1)):
        if use_nonterminal_only:
            rules = rule_vocab.get_list_of_applicable_nonterminal_rules(G)
        else: rules = rule_vocab.get_list_of_applicable_rules(G)
        if len(rules)>0:
            rule = rule_vocab.get_rule(rules[np.random.choice(len(rules))])
            G.apply_rule(rule)
        else: break
    rule_vocab.make_graph_terminal(G)
    return G



