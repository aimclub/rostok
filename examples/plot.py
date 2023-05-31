import matplotlib.pyplot as plt
from pathlib import Path
import pychrono as chrono
from rostok.utils.pickle_save import load_saveable
from rostok.virtual_experiment.sensors import DataStorage
import numpy as np
robot_joint_data: DataStorage = load_saveable(Path(r"results\Reports_23y_05m_31d_16H_42M\Robot_data.pickle")).get_data("joint_trajectories")
fig = plt.figure(figsize=(12, 5))
i = 1
time_list = list(np.linspace(0, 10, 10001))
a = chrono.ChFunction_Const(0)
reference_functions = [chrono.ChFunction_Const(0), chrono.ChFunction_Const(0),chrono.ChFunction_Sine(0.1, 0.5, 0.5),chrono.ChFunction_Const(0),chrono.ChFunction_Const(0),
                       chrono.ChFunction_Const(0), chrono.ChFunction_Const(0),chrono.ChFunction_Sine(0.1, 0.5, 0.5),chrono.ChFunction_Const(0),chrono.ChFunction_Const(0)]
reference_array =[]
for function in reference_functions:
    one_function_array = []
    for time in time_list:
        one_function_array.append(function.Get_y(time))
    reference_array.append(one_function_array)

for idx, data in robot_joint_data.items():
    fig.add_subplot(2, 5, i)
    plt.plot(time_list, data)
    plt.plot(time_list, reference_array[i-1])
    i+=1
plt.suptitle('joints')
plt.show()
print()