from copy import deepcopy

import networkx as nx
from matplotlib import pyplot as plt

from rostok.graph_generators.graph_reward import Reward
from rostok.graph_grammar.node import GraphGrammar, Rule
from rostok.graph_grammar.rule_vocabulary import RuleVocabulary
from rostok.trajectory_optimizer.control_optimizer import GraphRewardCalculator
from rostok.utils.states import RobotState


def rule_is_terminal(rule: Rule):
    """Function finding non terminal rules

    Args:
        rule (Rule): Input rule to checking

    Returns:
        int: amount of terminal nodes in rule
    """
    terminal_rule = [node[1]["Node"].is_terminal for node in rule.graph_insert.nodes.items()]
    return sum(terminal_rule)


class RuleAction:

    def __init__(self, rule):
        """Class action like rule grammar

        Args:
            rule (Rule): The rule defines action
        """
        self.__rule = rule

    @property
    def get_rule(self):
        return self.__rule

    def get_replaced_node(self):
        return self.__rule.replaced_node

    def is_terminal(self):
        return self.__rule.is_terminal

    # For this algorithm need override hash method

    def __hash__(self):
        return hash(self.__rule)

    def __eq__(self, __o) -> bool:
        if isinstance(__o, RuleAction):
            return self.__rule == __o.get_rule
        return False


class GraphEnvironment():

    def __init__(self, initilize_graph, rules, max_numbers_rules_non_terminal=20):
        """Class of "environment" of graph grammar

        Args:
            initilize_graph (GraphGrammar): Initial state of the graph
            rules (list[Rule]): List of rules
            max_numbers_rules_non_terminal (int): Max amount of non-terminal rules. Defaults to 20.
        """
        self.init_graph = deepcopy(initilize_graph)
        self.graph = deepcopy(initilize_graph)
        self._actions = [RuleAction(r) for r in rules] if rules is not None else None
        self.max_actions_not_terminal = max_numbers_rules_non_terminal
        self.current_player = 1
        self.reward = 0
        self.counter_action = 0

    # Need override for mcts libary
    def getCurrentPlayer(self):
        return self.current_player

    def getPossibleActions(self):
        """Getter possible actions for current state
        """

        def filter_exist_node(action):
            out = False
            flags_max_actions = self.counter_action >= self.max_actions_not_terminal
            if action.get_replaced_node().label == node:
                if action.is_terminal():
                    out = True
                if not flags_max_actions:
                    out = True

            return out

        label_nodes = {node[1]["Node"].label for node in self.graph.nodes.items()}
        possible_actions = []
        for node in label_nodes:
            possible_actions += set(filter(filter_exist_node, self._actions))

        return possible_actions

    def takeAction(self, action):
        """Take action and return new state environment

        Args:
            action (RuleAction): Action to take

        Returns:
            GraphEnvironment: New state environment after action taken
        """
        rule_action = action.get_rule
        new_state = deepcopy(self)
        new_state.graph.apply_rule(rule_action)
        if not action.is_terminal():
            new_state.counter_action += 1
        return new_state

    def isTerminal(self):
        """Condition on terminal graph

        Returns:
            bool: State graph is terminal
        """
        terminal_nodes = [node[1]["Node"].is_terminal for node in self.graph.nodes.items()]
        return sum(terminal_nodes) == len(terminal_nodes)

    # getter reward
    def getReward(self):
        """Reward in number (3) of nodes graph mechanism

        Returns:
            float: Reward of terminal state graph
        """
        nodes = [node[1]["Node"] for node in self.graph.nodes.items()]
        self.reward = 10 if len(nodes) == 4 else 0
        return self.reward

    def step(self, action: RuleAction, render=False):
        """Move current environment to new state

        Args:
            action (RuleAction): Action is take
            render (bool): Turn on render each step. Defaults to False.

        Returns:
            bool, GraphGrammar: Return state of graph. If it is terminal then finish generate graph
            and new state graph.
        """
        new_state = self.takeAction(action)
        self.graph = new_state.graph
        self.reward = new_state.reward
        self.counter_action = new_state.counter_action
        done = new_state.isTerminal()

        if render:
            plt.figure()
            nx.draw_networkx(self.graph,
                             pos=nx.kamada_kawai_layout(self.graph, dim=2),
                             node_size=800,
                             labels={n: self.graph.nodes[n]["Node"].label for n in self.graph})
            plt.show()
        return done, self.graph

    def reset(self, new_rules=None):
        """Reset environment (experimental version)

        Args:
            new_rules (list[Rule]): Replace on new rules. Defaults to None.
        """
        self.graph = self.init_graph
        self.reward = 0
        self.counter_action = 0
        if new_rules:
            self._actions = [RuleAction(r) for r in new_rules]

    def __eq__(self, __o) -> bool:
        if isinstance(__o, GraphEnvironment):
            is_graph_eq = __o.graph == self.graph
            return is_graph_eq
        return False


class GraphVocabularyEnvironment(GraphEnvironment):

    def __init__(self,
                 initilize_graph: GraphGrammar,
                 graph_vocabulary: RuleVocabulary,
                 optimizer: GraphRewardCalculator,
                 max_numbers_rules_non_terminal: int = 20):
        """Subclass graph environment on rule vocabulary instead rules and with real reward on
        simulation and control optimizing

        Args:
            initilize_graph (GraphGrammar): Initial state of the graph
            rule_vocabulary (RuleVocabulary): Object of the rule vocabulary for manipulation on
            graph
            max_numbers_rules_non_terminal (int): Max amount of non-terminal rules.
            Defaults to 20.
        """
        super().__init__(initilize_graph, None, max_numbers_rules_non_terminal)
        self.actions: RuleVocabulary = graph_vocabulary
        self.optimizer = optimizer
        self.state: RobotState = RobotState(graph_vocabulary)
        self.movments_trajectory = None

    def getPossibleActions(self):
        """Getter possible actions for current state
        """
        if self.counter_action < self.max_actions_not_terminal:
            possible_rules_name = self.actions.get_list_of_applicable_rules(self.graph)
        else:
            possible_rules_name = self.actions.get_list_of_applicable_terminal_rules(self.graph)

        possible_rules = [self.actions.rule_dict[str_rule] for str_rule in possible_rules_name]
        possible_actions = set(RuleAction(rule) for rule in possible_rules)
        return list(possible_actions)

    def getReward(self):
        result_optimizer = self.optimizer.start_optimisation(self.graph)
        self.reward = result_optimizer[0]
        self.movments_trajectory = result_optimizer[1]
        print(self.reward)
        return self.reward

    def takeAction(self, action):
        """Take action and return new state environment

        Args:
            action (RuleAction): Action to take

        Returns:
            GraphEnvironment: New state environment after action taken
        """
        rule_action = action.get_rule
        rule_dict = self.actions.rule_dict
        rule_name = list(rule_dict.keys())[list(rule_dict.values()).index(rule_action)]
        new_state = deepcopy(self)
        new_state.state.add_rule(rule_name)
        new_state.graph.apply_rule(rule_action)
        if not action.is_terminal():
            new_state.counter_action += 1
        return new_state

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ["actions", "optimizer"]:
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))

        return result


class GraphStubsEnvironment(GraphEnvironment):

    def __init__(self, initilize_graph, rules, max_numbers_rules_non_terminal=20):
        """Subclass graph environment for testing on stubs reward

        Args:
            initilize_graph (GraphGrammar): Initial state of the graph
            rules (list[Rule]): List of rules
            max_numbers_rules_non_terminal (int): Max amount of non-terminal rules. Defaults to 20.
        """
        super().__init__(initilize_graph, rules, max_numbers_rules_non_terminal)
        self.map_nodes_reward = {}

    def set_node_rewards(self, map_of_reward: map, func_reward: Reward.complex):
        self.map_nodes_reward = map_of_reward
        self.function_reward = func_reward

    def getReward(self):
        reward = self.function_reward(self.graph, self.map_nodes_reward)
        self.reward = reward
        return self.reward
