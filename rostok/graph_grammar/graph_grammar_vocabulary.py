from rostok.graph_grammar.graph_grammar import GraphGrammar, Rule
from rostok.graph_grammar.rule_vocabulary import RuleVocabulary


class GraphGrammarVocabulary(GraphGrammar):

    def __init__(self, rule_vocablary: RuleVocabulary, **attr):
        super().__init__(**attr)
        self.rule_vocabulary = rule_vocablary
        self.rule_list = []

    def apply_rule(self, rule: Rule):
        if isinstance(rule, Rule):
            self.rule_list.append("Rule_Object")
        else:
            self.rule_list.append(rule)
            rule = self.rule_vocabulary.get_rule(rule)
            if self.rule_vocabulary is None:
                raise Exception("Without rule vocabulary the rules must be Rule class objects.")

        if not rule.is_terminal:
            self.counter_nonterminal_rules += 1

        ids = self.find_nodes(rule.replaced_node)
        edge_list = list(self.edges)
        id_closest = self.closest_node_to_root(ids)
        if rule.graph_insert.order() == 0:
            # Stub removing leaf node if input rule is empty
            out_edges_ids_node = list(filter(lambda x: x[0] == id_closest, edge_list))
            if out_edges_ids_node:
                raise Exception("Trying delete not leaf node")
            self.remove_node(id_closest)
        else:
            self._replace_node(id_closest, rule)