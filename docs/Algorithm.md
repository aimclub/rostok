# Algorithms Description

The basis for implemented co-design algorithms is the graph representation of the mechanism. The nodes of the graphs are the representations of the parts of the mechanism.The graph made of such nodes unambiguously determine the physical properties of the mechanism. A graph can be generated from the starting point using the set of rules that guarantee that any final state is physically possible.

## Search for grasping mechanism with open kinematic chain

The algorithm generates the grasping mechanism for the user defined object using the set of user-determined rules. The graphs constructed by the set of rules form the space of possible solutions and the algorithm searches that space to obtain the best design. Currently, the rules are set to generate the open kinematic chain mechanisms.  

The ability of the generated mechanism/graph to grasp the object is simulated with the physical engine Pychrono. The simulation of the physical properties of the grasping process results in the scalar reward that is calculated using the four criterions:  
1. Criterion of isotropy of contact forces  
        Description: list cont contains mean values of contact force for each body element. If sum of cont is equal zero (no body is in contact), then criterion f1 = 0.  
        Otherwise, delta_u (standard deviation of contact forces) is calculated. We want to minimize deviation.  
2. Criterion of number of contact surfaces  
        Description: f2 is equal the ratio of mean value of contact surfaces (during simulation) to the overall potential number of contact surfaces.  
3. Criterion of mean values of distance between each of fingers  
        Description: algorithm has to minimize mean values of the distance for each finger.  
4. Criterion of simulation time  
        Description: algorithm has to maximize simulation time  

In addition to the design of the parts, the second essential part of the mechanism is the optimal trajectories of the actuators. For each design we optimize these  trajectories. The optimization is based on the same reward. Final reward for a design is calculated with the optimized trajectories and is used to search for the most effective design. 
The graph generating rules are fed to the searching algorithm to get the list of available actions at each step of the generation. Currently we use Monte Carlo Tree Search algorithm to explore the space of the possible designs. Therefore, at any step the algorithm gradually grow several designs and calculate their rewards in order to make a step in generation.     

The input data should include two different parts: 

1. The specification of the details for the mechanism:

* sizes of links, size of the palm and sizes of the fingertips
* amount of fingers (left and right) and possible angles in respect to the palm  

2. The shape of the object to grasp  

The output is the designed mechanism, in the graph form, the information about actuator trajectories and the obtained reward
The output can be used to start and visualize the simulation of the final design
The history of the exploration of the design space is saved within log file 
