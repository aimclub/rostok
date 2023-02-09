Installation guide
==================

* Create the environment using

  ``conda env create -f environment.yml``

* Activate the environment ``rostok``

  ``conda activate rostok``

* Install the lates version of PyChrono physical engine using

  ``conda install -c projectchrono pychrono``

* Install the package in development mode

  ``pip3 install -e .``

Known issues
------------

At some PC's one can see a problem with the tcl module ``version conflict for package "Tcl": have 8.6.12, need exactly 8.6.10``, try to install tk 8.6.10 using ``conda install tk=8.6.10``

After the installation of the package one can get an error ``Original error was: DLL load failed while importing _multiarray_umath: The specified module could not be found`` , try to reinstall numpy in the rostok environment
