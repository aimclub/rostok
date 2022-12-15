# Rostok framework for co-design of dynamical systems

Rostok is an open source library which provides the framework for generative co-design of mechatronic and robotic systems. Currently, the generative design is mainly used in production of static parts, while our project goal is to shift the generative design to the domain of the dynamic systems.  

The main feature of our framework is the ability to search for only physically possible designs. It is achieved by using the graph representation of the mechanisms and the constructing rules that gradually transform graph from base state to the final state staying within predetermined boundaries. Our library is a framework where user can specify building blocks and rules of graph generation for solving the co-design problems. For now, we implemented search for grasping  mechanism with open kinematic chain, but we plan to apply the idea of generative design to other problems in future.

 