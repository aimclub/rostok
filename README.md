# Rostok
Rostok is an open source Python module for co-design of grasping mechanisms. It provides an API to specify sizes of the detailes and constrcting rules for the mechanism as well as to set the object to grasp. The searching algoriphm takes the provided rules and object and search for the optimal design with the closed kinematic. The search is based on the Reinforcement Learning 

# Prerequisites
* Anaconda3 
* Usage of the Docker reqires installation of Х-server for Windows https://sourceforge.net/projects/vcxsrv/

![project_algorithm](docs/Algorithm_sceme.jpg)
# Installation
 
* Create the environment using `conda env create -f environment.yml`
* activate the environment `rostok`  
* Install the lates version of PyChrono physical engine using `conda install -c projectchrono pychrono`  
* Install the package in development mode `pip3 install -e . `  


# Running Examples
Open the directory with the ROSTOK installed  
`python3 app\app.py `  
The algorithm output is the graph representation of the found optimal design, the file `control.csv` with the optimized control data and the file `robot.jpg` with the picture of the generated grasping mechanism. 

# Issues
At some computers one can see a problem with the tcl module `version conflict for package "Tcl": have 8.6.12, need exactly 8.6.10`, try to install tk 8.6.10 using `conda install tk=8.6.10`

After the installation of the package one can get an error `Original error was: DLL load failed while importing _multiarray_umath: The specified module could not be found` , try to reinstall numpy in the rostok environment

## Acknowledgments
### Affiliation

The library was developed in [ITMO University](https://en.itmo.ru/).

### Developers


## Contacts
* ----- for collaboration suggestions
* Kirill Zharkov kzharkov@itmo.ru for technical questions



