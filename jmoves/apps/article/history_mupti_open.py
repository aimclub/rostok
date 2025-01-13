from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from auto_robot_design.optimization.analyze import get_optimizer_and_problem, get_pareto_sample_linspace, get_pareto_sample_histogram

optimizer, problem, res = get_optimizer_and_problem(
    "results\\multi_opti_preset2\\topology_0_2024-05-29_18-48-58")
sample_x, sample_F = get_pareto_sample_linspace(res, 10)
sample_x2, sample_F2 = get_pareto_sample_histogram(res, 10)
 


save_p = Path(str(PATH_CS) + "/" + "plots")
save_p.mkdir(parents=True, exist_ok=True)

history_mean = np.array(optimizer.history["Mean"])

plt.figure()

plt.figure()
plt.scatter(sample_F2[:, 0], sample_F2[:, 1])
plt.title("from res2")
plt.show()
