import numpy as np
from FEM2 import utils as MSA
import scipy as sp
import matplotlib.pyplot as plt

class Node:
    def __init__(self, coords: np.array, BCs: np.ndarray(6) == None, loads: np.zeros(6), id):
        
        self.coords = coords
        self.BCs = BCs
        self.loads = loads
        self.id = id

    def nodal_info(self):
        return self.coords, self.BCs, self.loads, self.id
    
class Element:
    def __init__(self, E, nu, A, Iy, Iz, J, coords1, coords2, id, v_temp: np.ndarray = None):
        self.E = E
        self.nu = nu
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.coords1 = coords1
        self.coords2 = coords2
        self.id = id
        self.v_temp = v_temp

    def L(self):
        return np.sqrt((self.coords2[0] - self.coords1[0]) ** 2 + (self.coords2[1] - self.coords1[1]) ** 2 + (self.coords2[2] - self.coords1[2]) ** 2)

    
    def local_K(self):
        return MSA.local_elastic_stiffness_matrix_3D_beam(self.E, self.nu, self.A, self.L(), self.Iy, self.Iz, self.J)

    def gamma(self):
        
        Gamma = MSA.rotation_matrix_3D(self.coords1[0], self.coords1[1], self.coords1[2], self.coords2[0], self.coords2[1], self.coords2[2], self.v_temp)
        return MSA.transformation_matrix_3D(Gamma)

    def global_K(self):
        return self.gamma().T @ self.local_K() @ self.gamma()

    def el_info(self):
        return self.global_K(), self.id
    
class Fem:
    def __init__(self, num_nodes, dof_per_node, K_el, bc, load, id):
        self.num_nodes = num_nodes
        self.dof_per_node = dof_per_node
        self.K_el = K_el
        self.bc = bc
        self.load = load
        self.id = id


    def connectivity(self):
        con  = []
        for i in range(len(self.id)):
            con.append(self.id[i])
        return np.array(con)

    def Big_K(self):
        connectivity = self.connectivity()
        K = np.zeros([self.num_nodes * self.dof_per_node, self.num_nodes * self.dof_per_node])
        for elem in range(connectivity.shape[0]):  # Loop over elements
            nodes = connectivity[elem]  # Get nodes for this element
            Kel = self.K_el[elem]
    
    # Compute global DOF indices for this element
            global_dof_indices = np.concatenate([
                np.arange(6 * nodes[0], 6 * nodes[0] + 6),  # DOFs of first node
                np.arange(6 * nodes[1], 6 * nodes[1] + 6)   # DOFs of second node
            ])
    
    # Assemble local stiffness matrix into the global stiffness matrix
            for p in range(12):  # Local row
                for q in range(12):  # Local column
                    global_p = int(global_dof_indices[p])  # Map local to global row
                    global_q = int(global_dof_indices[q]) # Map local to global column
            
                    K[global_p, global_q] += Kel[p, q]  # Assemble
        return K

    def BC_vec(self):
        vec = np.zeros(self.num_nodes * self.dof_per_node)
        for i in range(self.num_nodes):
            vec[6*i : 6*i + 6] = self.bc[i]
        return np.array([bool(x) for x in vec])

    def force_vec(self):
        vec = np.zeros(self.num_nodes * self.dof_per_node)
        for i in range(self.num_nodes):
            vec[6*i : 6*i + 6] = self.load[i]
        return vec
    def fem_info(self):
        return self.Big_K(), self.BC_vec(), self.force_vec()
    
class solver:
    def __init__(self, num_nodes, dof_per_node, K, BC, F):
        self.num_nodes = num_nodes
        self.dof_per_node = dof_per_node
        self.K = K
        self.BC = BC
        self.F = F

    def solve(self):
        u_total = np.zeros(self.num_nodes * self.dof_per_node)
        id = np.where(self.BC==True)[0]
        all_indices = np.arange(self.num_nodes * self.dof_per_node)
        missing_id = np.setdiff1d(all_indices, id)
        #deleting

        modified_k = np.delete(self.K, id, axis=0)
        modified_k = np.delete(modified_k, id, axis=1)
        modified_f = np.delete(self.F, id)

        u = np.linalg.solve(modified_k, modified_f)
        u_total[missing_id] = u

        forces = self.K @ u_total
        return u_total, forces, id, missing_id
    
#Get the displacement, forces in local coordinates

class G_to_L:
    def __init__(self, A, I_rho, coords1, coords2, id, k, u, v_temp: np.ndarray = None):
        self.coords1 = coords1
        self.coords2 = coords2
        self.I_rho = I_rho
        self.A = A
        self.k = k
        self.u = u
        self.id = id
        self.v_temp = v_temp
        self.U = np.concatenate((self.u[6*self.id[0]:6*self.id[0]+6], self.u[6*self.id[1]:6*self.id[1]+6]))
        
    def L(self):
        return np.sqrt((self.coords2[0] - self.coords1[0]) ** 2 + (self.coords2[1] - self.coords1[1]) ** 2 + (self.coords2[2] - self.coords1[2]) ** 2)


    def gamma(self):     
        Gamma = MSA.rotation_matrix_3D(self.coords1[0], self.coords1[1], self.coords1[2], self.coords2[0], self.coords2[1], self.coords2[2], self.v_temp)
        return MSA.transformation_matrix_3D(Gamma)

    def local_f(self):
        f = self.k @ self.U
        F = self.gamma() @ f
        return F

    def geo_stiffness(self):
        return MSA.local_geometric_stiffness_matrix_3D_beam(self.L(), self.A, self.I_rho, self.local_f()[6], self.local_f()[9], self.local_f()[4], self.local_f()[5], self.local_f()[10], self.local_f()[11])

    def global_Kg(self):
        return self.gamma().T @ self.geo_stiffness() @ self.gamma()

class Kg:
    def __init__(self, num_nodes, dof_per_node, K_g, id):
        self.num_nodes = num_nodes
        self.dof_per_node = dof_per_node
        self.K_g = K_g
        self.id = np.array(id)


    def connectivity(self):
        con  = []
        for i in range(len(self.id)):
            con.append(self.id[i])
        return np.array(con)

    def Big_Kg(self):
        connectivity = self.connectivity()
        KG = np.zeros([self.num_nodes * self.dof_per_node, self.num_nodes * self.dof_per_node])
        for elem in range(connectivity.shape[0]):  # Loop over elements
            nodes = connectivity[elem]  # Get nodes for this element
            Kg = self.K_g[elem]
    
    # Compute global DOF indices for this element
            global_dof_indices = np.concatenate([
                np.arange(6 * nodes[0], 6 * nodes[0] + 6),  # DOFs of first node
                np.arange(6 * nodes[1], 6 * nodes[1] + 6)   # DOFs of second node
            ])
    
    # Assemble local stiffness matrix into the global stiffness matrix
            for p in range(12):  # Local row
                for q in range(12):  # Local column
                    global_p = int(global_dof_indices[p])  # Map local to global row
                    global_q = int(global_dof_indices[q]) # Map local to global column
            
                    KG[global_p, global_q] += Kg[p, q]  # Assemble
        return KG
    
def hermite_shape_func_transverse(L,t,v1,v2,th_1,th_2):

    x = t * L
    N1 = 1 - 3*(x/L)**2 + 2*(x/L)**3
    N2 = x * (1-x/L)**2
    N3 = 3*(x/L)**2 - 2*(x/L)**3
    N4 = -(x**2/L)*(1-x/L)
    v = N1*v1 + N2*th_1 + N3*v2 + N4*th_2
    return v

def critical(k,kg, num_nodes, dof_per_node, id):
    eigvals,eigvecs = sp.linalg.eig(k,-kg)
    real_pos_mask = np.isreal(eigvals) & (eigvals > 0)
    filtered_eigvals = np.real(eigvals[real_pos_mask])
    filtered_eigvecs = eigvecs[:,real_pos_mask]
    sorted_inds = np.argsort(filtered_eigvals)
    filtered_eigvals = filtered_eigvals[sorted_inds]
    filtered_eigvecs = filtered_eigvecs[:,sorted_inds]

    p = filtered_eigvals[0]
    e_vec = filtered_eigvecs[0]
    u = np.zeros(num_nodes * dof_per_node)
    u[id] = e_vec
    return p, u

def buckled(u, coords, ID, scale:float=1.0,int_pts:int=20,axis_equal:bool=False):

    fig= plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,projection='3d')
    for i in ID:
        t_int = np.linspace(0,1,int_pts)
        u1 = u[6*i[0]:6*i[0]+6]
        u2 = u[6*i[1]:6*i[1]+6]
        X1 = coords[i[0]]
        X2 = coords[i[1]]
        L = np.linalg.norm(X1-X2)
        x_ref = [X1[0],X2[0]]
        y_ref = [X1[1],X2[1]]
        z_ref = [X1[2],X2[2]]
        gamma = MSA.rotation_matrix_3D(X1[0], X1[1], X1[2], X2[0], X2[1], X2[2])
        U1 = gamma @ u1[:3]
        U2 = gamma @ u2[:3]
        hermite_u = []
        hermite_v = []
        hermite_w = []
        for cur_t in t_int:
            u_ = hermite_shape_func_transverse(L,cur_t,U1[0],U2[0],u1[-1],u2[-1])
            v_ = hermite_shape_func_transverse(L,cur_t,U1[1],U2[1],u1[-3],u2[-3])
            w_ = hermite_shape_func_transverse(L,cur_t,U1[2],U2[2],u1[-2],u2[-2])
            hermite_u.append(u_)
            hermite_v.append(v_)
            hermite_w.append(w_)
        hermite_u = np.array(hermite_u)
        hermite_v = np.array(hermite_v)
        hermite_w = np.array(hermite_w)
        local_disps_all =  np.concatenate((hermite_u.reshape(-1,1),hermite_v.reshape(-1,1),hermite_w.reshape(-1,1)),axis=1)
        global_disps_all = gamma.T@local_disps_all.T

        x_buckled = np.linspace(X1[0],X2[0],int_pts)
        y_buckled = np.linspace(X1[1],X2[1],int_pts)
        z_buckled = np.linspace(X1[2],X2[2],int_pts)

        x_buckled += scale*global_disps_all[0,:]
        y_buckled += scale*global_disps_all[1,:]
        z_buckled += scale*global_disps_all[2,:]

        ax.plot(x_ref,y_ref,z_ref,c='b',linestyle='--',label='Reference',linewidth=2)

        ax.plot(x_buckled,y_buckled,z_buckled,c='r',label='Buckled',linewidth=2)

       # if element_id==0:
           # ax.legend()
        
        ax.set_title(f'Reference vs Buckled Configuration w/ scale={scale}',fontsize=20)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if axis_equal:
            ax.set_aspect('equal')