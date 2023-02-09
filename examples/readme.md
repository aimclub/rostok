# Examples

* `block_render_example.py` - Creation blocks from `block_builder` in Chrono simulator. Gives simulation loop with mechanism.
* `control_optimization_example.py` - Setup joint torque optimization for grab mechanism. Optimizer select const torque for minimize `criterion_calc` function. Gives optimal torque values and reward function value.
* `example_simstep.py` - Demonstrate using virtual experiment simulation module. Mechanism for simulate creates in example vocabulary.
* `example_vocabulary.py` - Needed for other examples, contains examples using `NodeVocabulary` and `RuleVocabulary`.
* `graph_manipulation.py` - Demonstrate filtration nodes of `GraphGrammar`
* `reward_grab_mechanism.py` - Demonstrate usage of `criterion` module
* `rule_apply_graph_example.py` - Demonstrate how to works `Rule` from `GraphGrammar` and how to create your own set of generation rules. Randomly selects rules and shows their impact on the graph.
