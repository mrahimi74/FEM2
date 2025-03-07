# Finite Element Method - Part1

[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/sandialabs/sibl#license)

![Codecov](https://codecov.io/gh/mrahimi74/FEM2/branch/main/graph/badge.svg)

[![tests](https://github.com/Lejeune-Lab-Graduate-Course-Materials/bisection-method/actions/workflows/tests.yml/badge.svg)](https://github.com/Lejeune-Lab-Graduate-Course-Materials/bisection-method/actions)
---

### Table of Contents
* [Getting Started](#gs)
* [FEM algorithm](#algo)
* [Conda environemnt, installation, and testing](#install)
* [Tutorial](#tutorial)

---

### Getting Started

To be written

---

### FEM Algorithm <a name="algo"></a>

The **FEM** is a numerical method to solve partial differential equations.

1. **Node class**:
   - You should enter the coordinates, boundary conditions and loading conditions.
   - Note that BCs is a 1*6 array containing True and False. True is for known and False for unknown DoFs.
   - Loads is also a 1*6 array, the first three of which are loads and the other are moments.
   - You should enter the id of that node which is an integer. (i.e 1 or 2 ...)
   - Finally, this class gives you the coordinates, Bcs, Loads and id of that specific node.
2. **Element class**:
   - First, you should enter the mechanical properties of the element.
   - You should also enter an array containing the id of both nodes in the element.
   - Using Element.el_info() method, you can get the stiffness matrix and the number of that element.
3. **Fem class$**:
   - Depending on how many elements you have, you should make an array for stiffness matrices of the elements, boundary conditions and loads of the nodes.
   - Finally, you need to creat an id array for the connectivity of elements.
   - The class gives the assembled stiffness matrix, BCs and Loading condition arrays of whole nodes.
4. **solver class**:
   - This gives the DoFs and Reaction forces of the nodes.
---

### Conda environment, install, and testing <a name="install"></a>

To install this package, please begin by setting up a conda environment:
```bash
conda create --name FEM-env python=3.12
```
Once the environment has been created, activate it:

```bash
conda activate FEM-env
```
Double check that python is version 3.12 in the environment:
```bash
python --version
```
Ensure that pip is using the most up to date version of setuptools:
```bash
pip install --upgrade pip setuptools wheel
```
Create an editable install of the bisection method code (note: you must be in the correct directory):
```bash
pip install -e .
```
Test that the code is working with pytest:
```bash
pytest -v --cov=FEM --cov-report term-missing
```
Code coverage should be 97%. Now you are prepared to write your own code based on this method and/or run the tutorial. 


If you would like the open `tutorial.ipynb` located in the `tutorial` folder as a Jupyter notebook in the browser, you might need to install Jupyter notebook in your conda environment as well:
```bash
pip install jupyter
```
```bash
cd tutorial/
```
```bash
jupyter notebook tutorial.ipynb
```
---

### Tutorial <a name="tutorial"></a>

---

#### **Examples**

After following the installation instructions above, it will be possible to run the tutorial examples contained in the `tutorial` folder.