from generation_pickuper_pipe import control_optimizer
from rostok.utils.pickle_save import load_saveable

PATH_TO_PICKLE_RESULT = "./results/Reports_23y_01m_20d_15H_36M/MCTS_data.pickle"
control_optimizer.is_visualize = True

loaded_mcts_result = load_saveable(PATH_TO_PICKLE_RESULT)

best_graph, reward, best_control = loaded_mcts_result.get_best_info()
loaded_mcts_result.draw_best_graph()
loaded_mcts_result.plot_means()
func_reward = control_optimizer.create_reward_function(best_graph)
res = -func_reward(best_control)
