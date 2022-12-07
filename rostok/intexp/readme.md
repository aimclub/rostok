# Interaction Explorer
  
:no_entry: Works are frozen :no_entry:  
  
The package implements the search for a reliable interaction between the gripper and the object. The grip and object are user defined. The result of the package is a list of possible poses from which the grip can take and hold a rigid body, as well as quantitative estimates based on analytical analysis.

## Task List   
- [X] First step:  
    - [X] Description of the testee object  
    - [X] Loading a volume mesh from an .obj file  
    - [X] Loading object parameters from .xml file  
    - [X] Loading user poses from .xml file  
    - [ ] Easy generation of grasping poses:  
        - [X] Random  
        - [ ] From point cloud  
        - [ ] From convex hull  
- [ ] Determination of the topology of the body: 
    - [ ] Reeb graph  
    - [ ] Body skeletonization  
- [ ] Gripper structure integration:  
    - [ ] Gripper mechanism integration in graph form
    - [ ] Extraction of geometric features from gripper
    - [ ] Determining the quality of the gripper position  

## Used extensions

* [PyChrono](https://projectchrono.org/pychrono/)
* [Open3D](http://www.open3d.org/)
* [lxml](https://lxml.de/)
* [SciPy](https://scipy.org/)
