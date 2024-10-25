This file briefly describes the software and how to install it.

This software written in python implements some visualization with a user-interface (using PyQt5) for a couple of machine learning models to be used on benchmarking data.  A video of its use is also provided.

There is a UI_requirements.txt that will install all the required dependencies.
Use it with pip in the following way:
pip install -r UI_requirements.txt

There are three main python files and these are:
1) main.py
2) UIMainWindow.py
3) MLFunctionsForUI.py

The widgets were created with Qt Creator and stored in form.ui which is called by UIMainWindow.py

The main file to run is called "main.py"

The code is documented with docstrings for every function so for help on any function, do "help(...)"


At first, you can follow what's done in the video, and then begin experimenting.

