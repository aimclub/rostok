from pathlib import Path
from statistics import mean

from rostok.graph_generators.graph_reward import Reward
from rostok.graph_grammar.node import *
from rostok.graph_grammar.rule_vocabulary import RuleVocabulary
from rostok.trajectory_optimizer.control_optimizer import ControlOptimizer
from rostok.utils.pickle_save import OptimizedGraphReport, Saveable
from rostok.utils.states import *


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


class MCTSHelper(Saveable):
    optimizer = None
    @classmethod
    def set_optimizer(cls, opt):
        cls.optimizer = opt

    def __init__(self, rule_vocabulary, optimizer, path = Path("./result")) -> None:
        super().__init__(path, 'MCTS_data')
        self.actions: RuleVocabulary = rule_vocabulary
        self.step_counter = 0
        self.seen_graphs: OptimizedGraphReport = OptimizedGraphReport(path)
        self.seen_states: list[MCTSOptimizedState] = []
        MCTSHelper.set_optimizer(optimizer)
        self.main_state = RobotState(rules=rule_vocabulary)
        self.main_simulated_state = OptimizedState(self.main_state, 0, None)
        self.best_simulated_state = OptimizedState(self.main_state, 0, None)

    def convert_control_to_list(self, control):
        if control is None:
            control = []
        elif isinstance(control, (float,int)):
            control = [control]

        return list(control)

    def set_best_state(self, state, reward, control):
        """Set the values for best state

        Args:
            state (RobotState): the state of the best design
            reward (float): the best reward obtained during MCTS search
            control: parameters of the control for best design"""

        control =  self.convert_control_to_list(control)
        self.best_simulated_state = OptimizedState(state, reward, control)

    def set_main_optimized_state(self, state, reward, control):
        """Set the values for main state

        Args:
            state (RobotState): the state of the main design
            reward (float): the main reward obtained during MCTS search
            control: parameters of the control for main design"""

        control =  self.convert_control_to_list(control)
        self.main_simulated_state = OptimizedState(state, reward, control)

    def check_graph(self, new_graph):
        """Check if the graph is already in seen_graphs

        Args:
            new_graph: the graph to check"""

        if len(self.seen_graphs.graph_list) > 0:
            for optimized_graph in self.seen_graphs.graph_list:
                if optimized_graph.graph == new_graph:
                    reward = optimized_graph.reward
                    control = optimized_graph.control
                    print('seen reward:', reward)
                    return True, reward, control

        return False, 0, []

    def add_state(self, state: RobotState, reward: float, control):
        """Add a state, reward and control to current_rewards

        state: a new state to add
        reward: a new calculated reward
        control: control parameters for the new state
        """
        control =  self.convert_control_to_list(control)
        new_optimized_state = MCTSOptimizedState(state, reward, control, self.step_counter)
        self.seen_states.append(new_optimized_state)
        if reward > self.best_simulated_state.reward:
            self.set_best_state(state, reward, control)

    def get_best_info(self):
        """Get graph, reward and control for the best state"""
        graph = self.best_simulated_state.state.make_graph()
        return graph, self.best_simulated_state.reward, self.best_simulated_state.control

    def step(self, state, action: RuleAction):
        """Move current environment to new state

        Args:
            action (RuleAction): Action is take
            render (bool): Turn on render each step. Defaults to False.

        Returns:
            bool, GraphGrammar: Return state of graph. If it is terminal then finish generate
            graph and new state graph.
        """

        rule_action = action.get_rule
        rule_dict = self.actions.rule_dict
        rule_name = list(rule_dict.keys())[list(rule_dict.values()).index(rule_action)]
        self.main_state.add_rule(rule_name)
        new_state = state.takeAction(action)
        self.step_counter += 1
        done = new_state.isTerminal()
        if done:
            main_reward = new_state.getReward()
            main_control = new_state.movments_trajectory
            self.set_main_optimized_state(new_state, main_reward, main_control)

        return done, new_state

    def plot_means(self):
        """Plot the mean rewards for steps of MCTS search"""

        rewards = []
        for state in self.seen_states:
            i = state.step
            if len(rewards) == i:
                rewards.append([state.reward])
            else:
                rewards[i].append(state.reward)

            mean_rewards = [mean(on_step_rewards) for on_step_rewards in rewards]

        plt.figure()
        plt.plot(mean_rewards)
        plt.show()


class GraphVocabularyEnvironment(GraphEnvironment):

    def __init__(self,
                 initilize_graph: GraphGrammar,
                 helper: MCTSHelper,
                 max_numbers_rules_non_terminal=20):
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
        self.helper: MCTSHelper = helper
        self.state: RobotState = RobotState(helper.actions)
        self.movments_trajectory = None

    def getPossibleActions(self):
        """Getter possible actions for current state
        """
        if self.counter_action <= self.max_actions_not_terminal:
            possible_rules_name = self.helper.actions.get_list_of_applicable_rules(self.graph)
        else:
            possible_rules_name = self.helper.actions.get_list_of_applicable_terminal_rules(self.graph)

        possible_rules = [self.helper.actions.rule_dict[str_rule] for str_rule in possible_rules_name]
        possible_actions = set(RuleAction(rule) for rule in possible_rules)
        return list(possible_actions)

    def getReward(self):
        report = self.helper.check_graph(self.graph)
        if report[0]:
            self.reward = report[1]
            self.movments_trajectory = report[2]
            self.helper.add_state(self.state, self.reward, self.movments_trajectory)
            print('seen reward:', self.reward)
            return self.reward

        result_optimizer = self.helper.optimizer.start_optimisation(self.graph)    
        self.reward = - result_optimizer[0]
        self.movments_trajectory = result_optimizer[1]
        self.helper.seen_graphs.add_graph(self.graph, self.reward, self.movments_trajectory)
        self.helper.add_state(self.state, self.reward, self.movments_trajectory)
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
        rule_dict = self.helper.actions.rule_dict
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
            if k != "helper":
                setattr(result, k, deepcopy(v, memo))
            else: 
                setattr(result, k, v)
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

