# Algorithm Desription
The algorithm generates the grasping mechanism for the user defined object using the parts with sizes defined by the user.
The base of our algorithm is the graph representation of the mechanism. The graph made of special nodes unambiguasly determine the physical propeties of the mechanism. A mechanism graph can be generated from the starting point using the set of rules that guranttes that at any final state is physically possible. The nodes are the representaions of the parts of the mechanism. The graphs constructed by the set of rules form the space of possible solutions and the algorithm seraches that space to obtain the best design. Currently, the rules are set to generate the open kinematic chain mechanisms.

The ability of the generated mechanism/graph to grasp the object is simulated with the physical engine Pychrono. The simulation of the physical properties of the grasping process results in the scalar reward that is calculated using the four criterinons:
    1) Criterion of isotropy of contact forces
        Desciption: list cont contains mean values of contact force for each body element. If sum of cont is equal zero (no body is in contact), then criterion f1 = 0. 
        Otherwise, delta_u (standart deviation of contact forces) is calculated. We want to minimize deviation.
    2) Criterion of number of contact surfaces
        Desciption: f2 is equal the ratio of mean value of contact surfaces (during simulation) to the overall potentional number of contact surfaces.
    3) Criterion of mean values of distance between each of fingers
        Desciption: algorithm has to minimize mean values of the distance for each finger.
    4) Criterion of simulation time
        Desciption: algorithm has to maximize simulation time

In addition to the design of the parts, the second essential part of the mechanism is the optimal trajectories of the actuators. For each design we optimize these  trajectories. The optimization is also based on the same reward. Final reward for a design is calculated with the optimized trajectoris and is used to search for the most effective design. 
The graph generating rules are fed to the searching algorithm to get the list of availible actions at each step of the generation. Currently we use Momte Carlo Tree Search algorithm to explore the space of the possible designs. Therefore, at any step the algorithm gradually grow several designs and calculate their rewards in order to make a step in generation.     

The input data chould include two different parts:
1) The specification of the details for the mechanism:  
* sizes of links, size of the palm and sizes of the fingertips
* amount of fingers (left and right) and possible angles in respect to the palm
2) The shape of the object to grasp

The outpus if the designed mechanism, in the graph form, the information about actuator trajectories and the obtained reward
The output can be used to start and visualize the simulation of the final design
The history of the exploration of the design space is saved within log file 
