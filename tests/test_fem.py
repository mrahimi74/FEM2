import pytest
import numpy as np
from FEM2 import fem
import matplotlib.pyplot as plt
from FEM2 import utils as MSA


def test_node():
    coords = np.array([1.0, 2.0, 3.0])
    BCs = np.array([1, 0, 0, 1, 0, 1])
    loads = np.array([0, 0, -10, 0, 0, 0])
    node = fem.Node(coords, BCs, loads, id=1)
    assert node.nodal_info() == (coords, BCs, loads, 1)

def test_element():
    E, nu, A, Iy, Iz, J = 210e9, 0.3, 0.02, 1e-6, 1e-6, 5e-6
    coords1, coords2 = np.array([0, 0, 0]), np.array([1, 1, 1])
    v_temp = np.array([0, 1, 0])
    elem = fem.Element(E, nu, A, Iy, Iz, J, coords1, coords2, id=1, v_temp=v_temp)
    
    assert np.isclose(elem.L(), np.sqrt(3))
    assert isinstance(elem.local_K(), np.ndarray)
    assert isinstance(elem.global_K(), np.ndarray)
    assert elem.el_info()[1] == 1

def test_fem():
    num_nodes, dof_per_node = 2, 6
    K_el = [np.eye(12)]
    bc = [np.array([1, 1, 0, 0, 0, 0]), np.array([0, 0, 1, 1, 0, 0])]
    load = [np.zeros(6), np.array([0, 0, -10, 0, 0, 0])]
    id = [[0, 1]]
    fem_model = fem.Fem(num_nodes, dof_per_node, K_el, bc, load, id)
    
    assert np.array_equal(fem_model.connectivity(), np.array(id))
    assert isinstance(fem_model.Big_K(), np.ndarray)
    assert isinstance(fem_model.BC_vec(), np.ndarray)
    assert isinstance(fem_model.force_vec(), np.ndarray)

    fem_info = fem_model.fem_info()
    assert isinstance(fem_info, tuple)
    assert len(fem_info) == 3
    assert isinstance(fem_info[0], np.ndarray)  # Big_K
    assert isinstance(fem_info[1], np.ndarray)  # BC_vec
    assert isinstance(fem_info[2], np.ndarray)  # force_vec

def test_solver():
    num_nodes, dof_per_node = 2, 6
    K = np.eye(12) * 1000
    BC = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], dtype=bool)
    F = np.array([0, 0, -10, 0, 0, 0, 0, 0, -10, 0, 0, 0])
    
    fem_solver = fem.solver(num_nodes, dof_per_node, K, BC, F)
    u_total, forces,_,_ = fem_solver.solve()
    
    assert isinstance(u_total, np.ndarray)
    assert isinstance(forces, np.ndarray)
    assert u_total.shape == (num_nodes * dof_per_node,)
    assert forces.shape == (num_nodes * dof_per_node,)


def test_G_to_L():
    u = np.zeros(12)
    v_temp = np.array([0, 0, 1])
    obj = fem.G_to_L(0.01, 1e-6, np.array([0, 0, 0]), np.array([1, 0, 0]), [0, 1], np.eye(12), u, v_temp)
    
    assert np.isclose(obj.L(), 1.0)
    assert obj.gamma().shape == (12, 12)
    assert obj.local_f().shape == (12,)
    assert obj.geo_stiffness().shape == (12, 12)
    assert obj.global_Kg().shape == (12, 12)

def test_Kg():
    id_list = [[0, 1], [1, 2]]
    K_g = [np.eye(12), 2 * np.eye(12)]
    obj = fem.Kg(3, 6, K_g, id_list)
    
    assert obj.connectivity().shape == (2, 2)
    assert obj.Big_Kg().shape == (18, 18)

def test_hermite_shape_func_transverse():
    v = fem.hermite_shape_func_transverse(2, 0.5, 1, 2, 0.1, 0.2)
    assert isinstance(v, float)

def test_critical():
    k = np.array([[ 22.66772197,  30.12309833,   0.        ,   0.        ,
          0.        ,   1.50796447],
       [ 30.12309833,  40.23952933,   0.        ,   0.        ,
          0.        ,  -1.13097336],
       [  0.        ,   0.        ,   0.07539822,  -1.50796447,
          1.13097336,   0.        ],
       [  0.        ,   0.        ,  -1.50796447,  44.56228349,
        -24.35942611,   0.        ],
       [  0.        ,   0.        ,   1.13097336, -24.35942611,
         30.35261825,   0.        ],
       [  1.50796447,  -1.13097336,   0.        ,   0.        ,
          0.        ,  62.83185307]]) 
    kg = np.array([[-2.25600000e-02,  1.92000000e-03,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00, -8.00000000e-02],
       [ 1.92000000e-03, -2.14400000e-02,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  6.00000000e-02],
       [ 0.00000000e+00,  0.00000000e+00, -2.40000000e-02,
         8.00000000e-02, -6.00000000e-02,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  8.00000000e-02,
        -4.27026667e+00,  3.19520000e+00,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00, -6.00000000e-02,
         3.19520000e+00, -2.40640000e+00,  0.00000000e+00],
       [-8.00000000e-02,  6.00000000e-02,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00, -6.66666667e+00]])  # Example geometric stiffness
    
    p, u = fem.critical(k, kg, 2, 6, np.array([ 6,  7,  8,  9, 10, 11]))  # Using a valid test case
    
    assert u.shape == (12,), "Displacement vector shape mismatch"
    assert p > 0, "Eigenvector should not be zero"  # Ensure it's a valid buckling load


def test_buckled():
    u = np.zeros(12)
    coords = np.array([[0, 0, 0], [1, 0, 0]])
    ID = [[0, 1]]
    
    try:
        fem.buckled(u, coords, ID, scale=1.0, int_pts=10, axis_equal=False)
        assert True  # If no exception occurs, the test passes
    except Exception as e:
        assert False, f"buckled function raised an exception: {e}"