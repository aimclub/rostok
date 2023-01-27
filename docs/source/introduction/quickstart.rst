==========
Quickstart
==========

How to run generating algorithm of open chain grab mechanism
============================================================

The :py:module:module:`launch` has a :py:class:`OpenChainGen` class. The class controls the configuration and launching of the algorithm for generating the capture mechanism. 
You can customize all the parts yourself, or you can use the configuration file to create a ready-made class to run the basic algorithm. By default, the configuration file is in module :py:module:`launch`.
Next, the second module is described.

*Step 1. Creating config file

The .ini configuration file consists of five parts: Flats, Links, OptimizingControl, StopFlagSimulation and MCTS.

The Flats part has arries `width` to set the width of base palm flat.

``[Flats]
width=0.7, 0.9, 1.5 ``

In link part is setted possible length of links grab mechanism

``[Links]
length=0.4, 0.6, 0.8``

OptimizingControl is used to set hyperparameters for control optimization 

``[OptimizingControl]
gait = 2.5 ; 
up_bound = 10 ; Maximum input value for the joint
low_bound = 0 ; Minimum input value for the joint
iteration = 5 ; Number of optimization iterations. 
time_step = 0.001 ; Simulation time step 
weights = 3, 1, 1, 0 ; Criterion function weights``

StopFlagSimulation defines the conditions for stopping the simulation. The variable `flags` has the name of the stopping conditions.
So, there are possible flags:
- MaxTime: sets the maximum simulation time (mandatory flag). The variable `time_sim` is set for it.
- NotContact: Condition for stopping the simulation if the mechanism has not contacted the object during the time specified by the variable `time_with_no_contact`.
- Slipout: Condition for the object to slip out of the mechanism. The variables `time_slipout_error` and `time_with_no_contact` must be set to be used.

``[StopFlagSimulation]
flags = MaxTime, Slipout, NotContact
time_sim = 3
time_with_no_contact = 0.6
time_slipout_error = 0.1``

Setting the hyperparameters of the MCTS algorithm

``[MCTS]
iteration = 5 ; Number of iterations of the exploration.  
max_non_terminal_rules = 4 ; Maximum number of non-terminal rule application.``

Function :py:function:`create_generator_by_config`  create :py:class:`OpenChainGen` from config file.

.. code-block::python

    from rostok.launch.open_chain_gen import create_generator_by_config

    model: OpenChainGen = create_generator_by_config("rostok/launch/config.ini")

*Step 2. Setting a grab object

Moule :py:module:envbody_shapes has available shapes of object.

.. code-block::python

    from rostok.block_builder.envbody_shapes import Sphere

    model.set_grasp_object(Sphere())

*Step 3. Run algorithm

.. code-block::python
    reporter: MCTSSaveable = model.run_generation()

Method returns :py:class:`MCTSSaveable` object containing main and best graph. Class has method to save result and visualise best solution.
Instead of, :py:class:`OpenChainGen` has methods `save_result()` and `visualize_result()`

Example config file `config.ini`
::
    [Flats]
    width=0.7, 0.9, 1.5

    [Links]
    length=0.4, 0.6, 0.8

    [OptimizingControl]
    gait = 2.5
    up_bound = 10
    low_bound = 0
    iteration = 5
    time_step = 0.001
    weights = 3, 1, 1, 0

    [StopFlagSimulation]
    flags = MaxTime, Slipout, NotContact
    time_sim = 3
    time_with_no_contact = 0.6
    time_slipout_error = 0.1

    [MCTS]
    iteration = 5
    max_non_terminal_rules = 4

Script to run and save the solution
.. code-block::python
    from rostok.block_builder.envbody_shapes import Sphere
    from rostok.launch.open_chain_gen import create_generator_by_config

    model = create_generator_by_config("rostok/launch/config.ini")
    model.set_grasp_object(Sphere())
    reporter = model.run_generation()
    reporter.get_best_info()
    model.save_result()
    model.visualize_result()

