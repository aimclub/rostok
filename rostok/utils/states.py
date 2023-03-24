from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from rostok.graph_grammar.graph_grammar import GraphGrammar
from rostok.graph_grammar.rule_vocabulary import RuleVocabulary


# State here is a list of rules that can be used to create robots
class RobotState():
    """The class that represents a mechanism within a given rule set.

    This class represents a mechanism through the ordered list of rules that is used to
    build a graph. It requires a RuleVocabulary object that controls the building of the
    graph from state. The State can represent the unfinished design.

    Attributes:
        rule_list (list[str]): the list of rules that determines the state

        rules (RuleVocabulary): the rule set that to use for the state
    """

    def __init__(self, rules: RuleVocabulary, rule_list: list[str] = []):
        """Create an empty state or a state from the given rule list.

        The in list is copied and can be modified outside the state object.

        Args:
            rules (RuleVocabulary): the rule set for the state.
            rule_list (list[str]): the rule list to create a new state.

        Raises:
            Exception: one of rules in the rule_list does not belong to given the rule vocabulary.
        """

        self.__rules = rules
        list_of_rules = rules.rule_dict.keys()
        for rule in rule_list:
            if rule not in list_of_rules:
                raise Exception("Attempt to add a rule absent in the rule_vocabulary to a state")
        self.rule_list: list[str] = rule_list.copy()

    def add_rule(self, rule: str):
        """Add a rule to a list of the current state.

        Args:
            rule (str): the new rule to add to the state

        Raises:
            Exception: the rule does not belong to the rule vocabulary assigned to the state
        """

        if rule not in self.__rules.rule_dict.keys():
            raise Exception("Attempt to add to a state a rule that is absent in the" +
                            "rule_vocabulary")

        self.rule_list.append(rule)

    def create_and_add(self, rule: str):
        """Create a new state and add rule to it, return the new state.

        Args:
            rule (str): the new rule to add to the state

        Raises:
            Exception: the rule does not belong to the rule vocabulary assigned to the state
        """

        if rule not in self.__rules.rule_dict.keys():
            raise Exception("Attempt to add to a state a rule that is absent in the" +
                            "rule_vocabulary")

        new = RobotState(self.__rules, self.rule_list)
        new.add_rule(rule)
        return new

    def make_graph(self):
        """Return a GraphGrammar object built with the rule list and rule vocabulary"""

        graph = GraphGrammar()
        for rule in self.rule_list:
            graph.apply_rule(self.__rules.get_rule(rule))

        self.__rules.make_graph_terminal(graph)
        return graph

    @property
    def rules(self):
        return self.__rules

    def __hash__(self):
        answer = ''
        answer = answer.join(self.rule_list)
        return hash(answer)


@dataclass
class OptimizedState():
    """Class that represents the state with a calculated reward

    Attributes:
        state (RobotState): the state to calculate reward
        reward (float): the calculated reward of the state
        control (list[float]): the parameters of the optimized control"""

    state: RobotState
    reward: float
    control: Any


@dataclass
class MCTSOptimizedState():
    """Class that represents the state with a calculated reward and has the step number

    Attributes:
        state (RobotState): the state to calculate reward
        reward (float): the calculated reward of the state
        control (list[float]): the parameters of the optimized control
        step (int): MCTS step this state was obtained at"""

    state: RobotState
    reward: float
    control: Any
    step: int


class OptimizedGraph():
    """Class that represents the graph with a calculated reward
    
        Attributes:
        graph (RobotState): the graph to calculate reward
        reward (float): the calculated reward of the state
        control (list[float]): the parameters of the optimized control
    """

    def __init__(self, graph, reward, control):
        self.__graph = deepcopy(graph)
        self.__reward = reward
        self.__control = deepcopy(control)

    @property
    def graph(self):
        return self.__graph

    @property
    def reward(self):
        return self.__reward

    @property
    def control(self):
        return self.__control
