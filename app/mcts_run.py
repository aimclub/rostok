import mcts_runner

from tendon_graph_evaluators import evaluator_tendon_standart, evaluator_tendon_standart_parallel
from tendon_graph_evaluators import mcts_hyper_default, evaluator_tendon_fast_debug
import tendon_driven_cfg
from rostok.library.rule_sets.rulset_simple_fingers import create_rules

if __name__ == "__main__":
    rules_vocab = create_rules()
    mcts_runner.run_mcts(rules_vocab, evaluator_tendon_standart_parallel, mcts_hyper_default)