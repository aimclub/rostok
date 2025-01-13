

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx




MARKERS = ["o", "s", "*", "p", "D", "^", "v", "X"]

def prepare_data_to_visualize(data, problem):
    """
    Prepare data to visualize
    """
    num_joints = len(problem.opt_joints)
    
    feature_jps = np.reshape(data["X"], (-1 , len(data["X"][0]) // num_joints))
    
    costs = np.array([fs for fs in data["Fs"] for __ in range(num_joints)])
    
    total_cost = np.array([f for f in data["F"] for __ in range(num_joints)])
    
    # Return data to visualize
    return feature_jps, costs, total_cost

def prepare_data_to_visualize_separeate_jps(data, problem):
    """
    Prepare data to visualize
    """
    num_joints = len(problem.opt_joints)
    num_features = len(data["X"][0]) // num_joints
    dataX = np.array(data["X"])
    features = []
    for id in range(0, len(dataX[0]), num_features):
        features.append(dataX[:,id:id+num_features])
    
    costs = np.array(data["Fs"])
    
    total_cost = np.array(data["F"])
    
    # Return data to visualize
    return features, costs, total_cost


def draw_jps_cost_on_graph(feature, cost, problem, marker = None):
    """
    Draw the cost of each joint on the graph
    """
    if marker is None:
        marker = "o"
    plt.scatter(feature[:,0], feature[:,1], c=cost, marker=marker)
    return plt


def draw_jps_distribution(feature):
    """
    Draw the cost of each joint on the graph
    """
    plt.hexbin(feature[:,0], feature[:,1], gridsize=50, cmap="YlOrBr")
    return plt

def draw_costs(cost1, cost2):
    
    plt.scatter(cost1, cost2)
    # plt.xlim(min(cost1)-0.01, max(cost1)+0.01)
    # plt.ylim(min(cost2)-0.1, max(cost2)+0.1)
    return plt