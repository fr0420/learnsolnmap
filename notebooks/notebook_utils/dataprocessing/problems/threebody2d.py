import os
import numpy as np 
import pandas as pd
from ..core import States, Trajectory, Dataset
from .base import BaseProblem


REF_TRAJ_FILEPATHS = {
    1: {
        "filepath": "/workspace/projects_rui/learnsolnmap/out/3body-2d/1/202411101626/ref/k=0/u.csv",
        "dt": 1e-2,
    },
    2: {
        "filepath": "/workspace/projects_rui/learnsolnmap/out/3body-2d/2/202411101911/ref/k=0/u.csv",
        "dt": 1e-2,
    },
    3: {
        "filepath": "/workspace/projects_rui/learnsolnmap/out/3body-2d/3/202411102337/ref/k=0/u.csv",
        "dt": 1e-2,
    },
    4: {
        "filepath": "/workspace/projects_rui/learnsolnmap/out/3body-2d/4/202411110023/ref/k=0/u.csv",
        "dt": 1e-2,
    },

    # 5: "/workspace/projects_rui/learnsolnmap/out/3body-2d/202411101709/ref/k=0/u.csv",
    # 6: "/workspace/projects_rui/learnsolnmap/out/3body-2d/202411101806/ref/k=0/u.csv",
    # 7: "/workspace/projects_rui/learnsolnmap/out/3body-2d/202411101834/ref/k=0/u.csv",
    # 8: "/workspace/projects_rui/learnsolnmap/out/3body-2d/202411101852/ref/k=0/u.csv",
}


EQUALMASS_REF_TRAJ_FILEPATHS = {
    1: {
        "filepath": "/workspace/projects_rui/learnsolnmap/out/3body-2d-equalmass/1/202411191201/ref/k=0/u.csv",  # KahanLi8
        "dt": 1e-2,
    },
    2: {
        "filepath": "/workspace/projects_rui/learnsolnmap/out/3body-2d-equalmass/2/202411191214/ref/k=0/u.csv",  # KahanLi8
        "dt": 1e-2,
    },
    3: {
        "filepath": "/workspace/projects_rui/learnsolnmap/out/3body-2d-equalmass/3/202412051622/ref/k=0/u.csv",  # DPRKN5
        "dt": 1e-2,
    },
    4: {
        "filepath": "/workspace/projects_rui/learnsolnmap/out/3body-2d-equalmass/4/202412081358/ref/k=0/u.csv",  # DPRKN5
        "dt": 1e-2,
    },
    5: {
        "filepath": "/workspace/projects_rui/learnsolnmap/out/3body-2d-equalmass/5/202412092355/ref/k=0/u.csv",  # DPRKN5
        "dt": 1e-2,
    },
    6: {
        "filepath": "/workspace/projects_rui/learnsolnmap/out/3body-2d-equalmass/6/202412161337/ref/k=0/u.csv",  # DPRKN5
        "dt": 1e-2,
    },
    7: {
        "filepath": "/workspace/projects_rui/learnsolnmap/out/3body-2d-equalmass/7/202412171119/ref/k=0/u.csv",  # DPRKN5
        "dt": 1e-2,
    },
}


def get_rotation_matrices(vectors, target_vector):
    """
    Returns the rotation matrices that align each vector in `vectors` to the `target_vector`.

    Parameters:
        vectors (np.ndarray): Array of shape (N, 2) representing the list of 2D vectors to align.
        target_vector (np.ndarray): A 1D array of shape (2,) representing the target vector.

    Returns:
        np.ndarray: Array of shape (N, 2, 2) containing rotation matrices.
    """
    # Normalize the input vectors
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    target_vector = target_vector / np.linalg.norm(target_vector)

    # Extract the components of the target vector
    tx, ty = target_vector

    # Compute the angles of rotation needed for each vector
    angles = np.arctan2(ty, tx) - np.arctan2(vectors[:, 1], vectors[:, 0])

    # Create rotation matrices for each angle
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    rotation_matrices = np.zeros((len(vectors), 2, 2))
    rotation_matrices[:, 0, 0] = cos_angles
    rotation_matrices[:, 0, 1] = -sin_angles
    rotation_matrices[:, 1, 0] = sin_angles
    rotation_matrices[:, 1, 1] = cos_angles

    return rotation_matrices


class ThreeBody2D(BaseProblem):

    def __init__(self, m1=100., m2=1., m3=0.001, G=1.):
        super().__init__(dof=6)
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.G = G
    
    def __eq__(self, other):
        if isinstance(other, ThreeBody2D):
            return self.m1 == other.m1 and self.m2 == other.m2 and self.m3 == other.m3 and self.G == other.G
        return NotImplemented
    
    def __repr__(self):
        return f"ThreeBody2D(m1={self.m1}, m2={self.m2}, m3={self.m3}, G={self.G})"
    
    def get_pq(self, u):
        v, x = np.split(u, 2, axis=-1)
        p1 = self.m1 * v[..., :2]
        p2 = self.m2 * v[..., 2:4]
        p3 = self.m3 * v[..., 4:]
        q1 = x[..., :2]
        q2 = x[..., 2:4]
        q3 = x[..., 4:]
        return (p1, p2, p3), (q1, q2, q3)
    
    def get_vx(self, u):
        v, x = np.split(u, 2, axis=-1)
        v1 = v[..., :2]
        v2 = v[..., 2:4]
        v3 = v[..., 4:]
        x1 = x[..., :2]
        x2 = x[..., 2:4]
        x3 = x[..., 4:]
        return (v1, v2, v3), (x1, x2, x3)
    
    def convert_pq_to_vx(self, pq):
        p, q = np.split(pq, 2, axis=-1)
        v1 = p[..., :2] / self.m1
        v2 = p[..., 2:4] / self.m2
        v3 = p[..., 4:] / self.m3
        x = q 
        return np.concatenate((v1, v2, v3, x), axis=-1)
    
    def convert_vx_to_pq(self, vx):
        v, x = np.split(vx, 2, axis=-1)
        p1 = self.m1 * v[..., :2]
        p2 = self.m2 * v[..., 2:4]
        p3 = self.m3 * v[..., 4:]
        q = x
        return np.concatenate((p1, p2, p3, q), axis=-1)
    
    def get_x_rotatingframe(self, u):
        _, x = np.split(u, 2, axis=-1)
        x1 = x[..., :2]
        x2 = x[..., 2:4]
        x3 = x[..., 4:]
        x2_0 = x2[0]
        rot = get_rotation_matrices(x2, x2_0)
        x1_rot = np.einsum("ijk,ik->ij", rot, x1)
        x2_rot = np.einsum("ijk,ik->ij", rot, x2)
        x3_rot = np.einsum("ijk,ik->ij", rot, x3)
        return x1_rot, x2_rot, x3_rot

    def compute_hamiltonian(self, u):
        return self.compute_potential_energy(u) + self.compute_kinetic_energy(u)
    
    def compute_potential_energy(self, u):
        x1, x2, x3 = self.get_vx(u)[1]
        U = -self.G * (self.m1 * self.m2 / np.linalg.norm(x1 - x2, axis=-1) \
                     + self.m1 * self.m3 / np.linalg.norm(x1 - x3, axis=-1) \
                     + self.m2 * self.m3 / np.linalg.norm(x2 - x3, axis=-1))
        return U
    
    def compute_kinetic_energy(self, u):
        v1, v2, v3 = self.get_vx(u)[0]
        K = 0.5 * (self.m1 * np.sum(v1**2, axis=-1) \
                 + self.m2 * np.sum(v2**2, axis=-1) \
                 + self.m3 * np.sum(v3**2, axis=-1))
        return K

    def compute_total_momentum(self, u):
        p1, p2, p3 = self.get_pq(u)[0]
        return p1 + p2 + p3
    
    def compute_total_angular_momentum(self, u):
        p1, p2, p3 = self.get_pq(u)[0]
        q1, q2, q3 = self.get_pq(u)[1]
        L1 = np.cross(q1, p1)
        L2 = np.cross(q2, p2)
        L3 = np.cross(q3, p3)
        return L1 + L2 + L3
    
    def compute_r12(self, u):
        x1, x2, _ = self.get_vx(u)[1]
        return np.linalg.norm(x2 - x1, axis=-1)
    
    def compute_r13(self, u):
        x1, _, x3 = self.get_vx(u)[1]
        return np.linalg.norm(x3 - x1, axis=-1)
    
    def compute_r23(self, u):
        _, x2, x3 = self.get_vx(u)[1]
        return np.linalg.norm(x3 - x2, axis=-1)
    
    def compute_du(self, u):
        v, x = np.split(u, 2, axis=-1)
        dv = self.compute_ddx(u)
        dx = v
        return np.concatenate((dv, dx), axis=-1)

    def compute_ddu(self, u):
        if u.ndim == 1:
            u = u.reshape(1, -1)
        jac = self.compute_jacobian(u) 
        du = self.compute_du(u)
        ddu = np.einsum('bij,bj->bi', jac, du)
        if len(u) == 1:
            ddu = ddu[0]
        return ddu
    
    def compute_ddx(self, u):
        x1, x2, x3 = self.get_vx(u)[1]
        r12 = np.linalg.norm(x2 - x1, axis=-1, keepdims=True)
        r13 = np.linalg.norm(x3 - x1, axis=-1, keepdims=True)
        r23 = np.linalg.norm(x3 - x2, axis=-1, keepdims=True)
        ddx1 = - self.G * (self.m2 * (x1 - x2) / r12**3 + self.m3 * (x1 - x3) / r13**3)
        ddx2 = - self.G * (self.m1 * (x2 - x1) / r12**3 + self.m3 * (x2 - x3) / r23**3)
        ddx3 = - self.G * (self.m1 * (x3 - x1) / r13**3 + self.m2 * (x3 - x2) / r23**3)
        return np.concatenate((ddx1, ddx2, ddx3), axis=-1)
    
    def compute_grad_H(self, u):
        """Compute dH/dp_i and dH/dq_i for i = 1, 2, 3."""
        (p1, p2, p3), (q1, q2, q3) = self.get_pq(u)
        r12 = np.linalg.norm(q2 - q1, axis=-1, keepdims=True)
        r13 = np.linalg.norm(q3 - q1, axis=-1, keepdims=True)
        r23 = np.linalg.norm(q3 - q2, axis=-1, keepdims=True)
        dHdq1 = self.G * self.m1 * (self.m2 * (q1 - q2) / r12**3 + self.m3 * (q1 - q3) / r13**3)
        dHdq2 = self.G * self.m2 * (self.m1 * (q2 - q1) / r12**3 + self.m3 * (q2 - q3) / r23**3)
        dHdq3 = self.G * self.m3 * (self.m1 * (q3 - q1) / r13**3 + self.m2 * (q3 - q2) / r23**3)
        grad_H = np.concatenate((p1 / self.m1, p2 / self.m2, p3 / self.m3, dHdq1, dHdq2, dHdq3), axis=-1)
        return grad_H
    
    def compute_hessian_H(self, u):
        """Compute the Hessian matrix of H with respect to p_i and q_i for i = 1, 2, 3."""
        q1, q2, q3 = self.get_pq(u)[1]
        batch_size = len(u)
        hessian_H = np.zeros((batch_size, 12, 12))

        r12 = np.linalg.norm(q2 - q1, axis=-1, keepdims=True)
        r13 = np.linalg.norm(q3 - q1, axis=-1, keepdims=True)
        r23 = np.linalg.norm(q3 - q2, axis=-1, keepdims=True)
        
        # momentum-momentum block (diagonal)
        hessian_H[:, :2, :2] = np.eye(2) / self.m1
        hessian_H[:, 2:4, 2:4] = np.eye(2) / self.m2
        hessian_H[:, 4:6, 4:6] = np.eye(2) / self.m3
        
        # function to compute the derivative of G * m_i * m_j (q_i - q_j) / |q_i - q_j|^3 with respect to q_i
        def compute_position_block(rij, qi, qj, mi, mj):
            
            r_inv_3 = 1 / (rij**3)
            r_inv_5 = 1 / (rij**5)
            
            diff = qi - qj
            outer = np.einsum('bi,bj->bij', diff, diff)
            
            I = np.eye(2)[None, :, :] * np.ones((batch_size, 1, 1))
            return self.G * mi * mj * (I * r_inv_3[:, None] - 3 * outer * r_inv_5[:, None])
    
        # position-position blocks
        # q1-q1 block
        hessian_H[:, 6:8, 6:8] = (
            compute_position_block(r12, q1, q2, self.m1, self.m2) +
            compute_position_block(r13, q1, q3, self.m1, self.m3)
        )
        
        # q2-q2 block
        hessian_H[:, 8:10, 8:10] = (
            compute_position_block(r12, q2, q1, self.m2, self.m1) +
            compute_position_block(r23, q2, q3, self.m2, self.m3)
        )
        
        # q3-q3 block
        hessian_H[:, 10:12, 10:12] = (
            compute_position_block(r13, q3, q1, self.m3, self.m1) +
            compute_position_block(r23, q3, q2, self.m3, self.m2)
        )
        
        # q1-q2 block
        q12_block = -compute_position_block(r12, q1, q2, self.m1, self.m2)
        hessian_H[:, 6:8, 8:10] = q12_block
        hessian_H[:, 8:10, 6:8] = q12_block
        
        # q1-q3 block
        q13_block = -compute_position_block(r13, q1, q3, self.m1, self.m3)
        hessian_H[:, 6:8, 10:12] = q13_block
        hessian_H[:, 10:12, 6:8] = q13_block
        
        # q2-q3 block
        q23_block = -compute_position_block(r23, q2, q3, self.m2, self.m3)
        hessian_H[:, 8:10, 10:12] = q23_block
        hessian_H[:, 10:12, 8:10] = q23_block
        
        return hessian_H

    def compute_jacobian(self, u):
        """Compute the Jacobian of the vector field (dp/dt, dq/dt) with respect to (p, q)."""
        hessian_H = self.compute_hessian_H(u)
        J_inv = np.zeros((12, 12))
        J_inv[:6, 6:] = -np.eye(6)
        J_inv[6:, :6] = np.eye(6)
        jacobian = np.einsum('ij,bjk->bik', J_inv, hessian_H)
        return jacobian

    def compute_stiffness(self, u, zero_tol=1e-6, ratio=False):
        """Compute the stiffness measure of the vector field (dp/dt, dq/dt) at the point u."""
        jacobian = self.compute_jacobian(u)
        stiffness_values = np.zeros(len(u))
        for i in range(len(u)):
            try:
                eigenvalues = np.linalg.eigvals(jacobian[i])
                eigenvalues_magnitudes = np.abs(eigenvalues)
                nonzero_vals = eigenvalues_magnitudes[eigenvalues_magnitudes > zero_tol]
                if ratio: 
                    stiffness_values[i] = np.max(nonzero_vals) / np.min(nonzero_vals)
                else:
                    stiffness_values[i] = np.max(nonzero_vals)
            except np.linalg.LinAlgError:
                stiffness_values[i] = np.nan
            except ValueError:
                stiffness_values[i] = np.nan
        return stiffness_values

    def compute_stiffness2(self, u, zero_tol=1e-14, ratio=False):
        """Compute the stiffness measure of the vector field (dp/dt, dq/dt) at the point u."""
        hessian_H = self.compute_hessian_H(u)
        matrices = hessian_H[:, :6, :6] @ hessian_H[:, 6:, 6:]
        stiffness_values = np.zeros(len(u))
        for i in range(len(u)):
            try:
                eigenvalues = np.linalg.eigvals(matrices[i])
                eigenvalues_magnitudes = np.abs(eigenvalues)
                nonzero_vals = eigenvalues_magnitudes[eigenvalues_magnitudes > zero_tol]
                sqrt_vals = np.sqrt(nonzero_vals)
                if ratio: 
                    stiffness_values[i] = np.max(sqrt_vals) / np.min(sqrt_vals)
                else:
                    stiffness_values[i] = np.max(sqrt_vals)
            except np.linalg.LinAlgError:
                stiffness_values[i] = np.nan
                print("LinAlgError")
            except ValueError:
                stiffness_values[i] = np.nan
                print("ValueError")
        return stiffness_values
    
    def compute_energies(self, u):
        return {
            "H": self.compute_hamiltonian(u),
            "U": self.compute_potential_energy(u),
            "K": self.compute_kinetic_energy(u),
        }
    
    def compute_errors(self, u, ref_u):

        # Compute trajectory errors
        pq = self.convert_vx_to_pq(u)
        ref_pq = self.convert_vx_to_pq(ref_u)
        diff_squares = (pq - ref_pq)**2
        abs_traj_err = np.sqrt(np.sum(diff_squares, axis=-1))
        abs_traj_err_p = np.sqrt(np.sum(diff_squares[:, :6], axis=-1))
        abs_traj_err_p1 = np.sqrt(np.sum(diff_squares[:, :2], axis=-1))
        abs_traj_err_p2 = np.sqrt(np.sum(diff_squares[:, 2:4], axis=-1))
        abs_traj_err_p3 = np.sqrt(np.sum(diff_squares[:, 4:6], axis=-1))
        abs_traj_err_q = np.sqrt(np.sum(diff_squares[:, 6:], axis=-1))
        abs_traj_err_q1 = np.sqrt(np.sum(diff_squares[:, 6:8], axis=-1))
        abs_traj_err_q2 = np.sqrt(np.sum(diff_squares[:, 8:10], axis=-1))
        abs_traj_err_q3 = np.sqrt(np.sum(diff_squares[:, 10:], axis=-1))
        rel_traj_err = abs_traj_err / np.linalg.norm(ref_pq, axis=-1)
        rel_traj_err_p = abs_traj_err_p / np.linalg.norm(ref_pq[:, :6], axis=-1)
        rel_traj_err_p1 = abs_traj_err_p1 / np.linalg.norm(ref_pq[:, :2], axis=-1)
        rel_traj_err_p2 = abs_traj_err_p2 / np.linalg.norm(ref_pq[:, 2:4], axis=-1)
        rel_traj_err_p3 = abs_traj_err_p3 / np.linalg.norm(ref_pq[:, 4:6], axis=-1)
        rel_traj_err_q = abs_traj_err_q / np.linalg.norm(ref_pq[:, 6:], axis=-1)
        rel_traj_err_q1 = abs_traj_err_q1 / np.linalg.norm(ref_pq[:, 6:8], axis=-1)
        rel_traj_err_q2 = abs_traj_err_q2 / np.linalg.norm(ref_pq[:, 8:10], axis=-1)
        rel_traj_err_q3 = abs_traj_err_q3 / np.linalg.norm(ref_pq[:, 10:], axis=-1)

        # Compute Hamiltonian errors 
        H = self.compute_hamiltonian(u)
        ref_H = self.compute_hamiltonian(ref_u)
        abs_H_err = np.abs(H - ref_H)
        rel_H_err = abs_H_err / np.abs(ref_H)
        
        # Compute total angular momentum errors
        Lz = self.compute_total_angular_momentum(u)
        ref_Lz = self.compute_total_angular_momentum(ref_u)
        abs_Lz_err = np.abs(Lz - ref_Lz)
        rel_Lz_err = abs_Lz_err / np.abs(ref_Lz)

        # Compute total momentum errors
        P = self.compute_total_momentum(u)
        ref_P = self.compute_total_momentum(ref_u)
        abs_P_err = np.linalg.norm(P - ref_P, axis=-1)
        rel_P_err = abs_P_err / np.linalg.norm(ref_P, axis=-1)

        return {
            "abs_traj_err": abs_traj_err, 
            "abs_traj_err_p": abs_traj_err_p,
            "abs_traj_err_p1": abs_traj_err_p1,
            "abs_traj_err_p2": abs_traj_err_p2,
            "abs_traj_err_p3": abs_traj_err_p3,
            "abs_traj_err_q": abs_traj_err_q,
            "abs_traj_err_q1": abs_traj_err_q1,
            "abs_traj_err_q2": abs_traj_err_q2,
            "abs_traj_err_q3": abs_traj_err_q3,
            "rel_traj_err": rel_traj_err,
            "rel_traj_err_p": rel_traj_err_p,
            "rel_traj_err_p1": rel_traj_err_p1,
            "rel_traj_err_p2": rel_traj_err_p2,
            "rel_traj_err_p3": rel_traj_err_p3,
            "rel_traj_err_q": rel_traj_err_q,
            "rel_traj_err_q1": rel_traj_err_q1,
            "rel_traj_err_q2": rel_traj_err_q2,
            "rel_traj_err_q3": rel_traj_err_q3,
            "abs_H_err": abs_H_err, 
            "rel_H_err": rel_H_err,
            "abs_Lz_err": abs_Lz_err,
            "rel_Lz_err": rel_Lz_err,
            "abs_P_err": abs_P_err,
            "rel_P_err": rel_P_err,
        }
    
    def compute_taylor_series(self, u0, t, order=1):
        """Compute the Taylor series expansion of the solution at time t with u(0)=u0."""
        # u0: (12,) or (batch_size, 12)
        # t: scalar or (n,) array
        # order: 0, 1, or 2
        # returns u: (n, 12) or (n, batch_size, 12)

        if isinstance(t, (int, float)):
            t = np.array([t])
        if not isinstance(t, np.ndarray):
            raise ValueError("t must be a scalar or a numpy array.")
        if t.ndim > 1:
            raise ValueError("t must be a scalar or a 1D array.")
        
        # reshape t for broadcasting
        t = t.reshape(t.shape + (1,) * u0.ndim)

        if order == 0:
            return u0[None, :].repeat(len(t), axis=0)
        elif order == 1:
            return u0 + t * self.compute_du(u0)
        elif order == 2:
            du = self.compute_du(u0)
            ddu = self.compute_ddu(u0)
            return u0 + t * du + 0.5 * t**2 * ddu
        else:
            raise ValueError("Only orders 0, 1, and 2 are supported.")
    
    @classmethod
    def get_reference_filepaths(cls, category='default'):
        """
        Get reference trajectory filepaths for ThreeBody2D problem.
        
        Parameters:
        -----------
        category : str, optional
            The category of reference trajectories to retrieve. Options:
            - 'default': Standard three-body problem trajectories
            - 'equalmass': Equal mass three-body problem trajectories
            
        Returns:
        --------
        dict
            Dictionary containing reference trajectory filepaths organized by
            initial condition indices.
        """
        if category == 'default':
            return REF_TRAJ_FILEPATHS
        elif category == 'equalmass':
            return EQUALMASS_REF_TRAJ_FILEPATHS
        else:
            raise ValueError(f"Unknown category '{category}'. Available categories: {cls.get_available_reference_categories()}")
    
    @classmethod
    def get_available_reference_categories(cls):
        """
        Get list of available reference trajectory categories for ThreeBody2D.
        
        Returns:
        --------
        list
            List of available category names for reference trajectories.
        """
        return ['default', 'equalmass']


class ThreeBody2DDataset(Dataset):

    @classmethod
    def load_from_file(cls, filepath, name="none", m1=100., m2=1., m3=0.001, G=1.):
        df = pd.read_csv(filepath)
        data = States(df.values, ThreeBody2D(m1, m2, m3, G))
        return cls(data, name)
    
    @classmethod
    def from_vx(cls, vx, name="none", m1=100., m2=1., m3=0.001, G=1.):
        return cls(States(vx, ThreeBody2D(m1, m2, m3, G)), name)
    

class ThreeBody2DTrajectory(Trajectory): 

    @classmethod
    def load_from_file(cls, filepath, dt=None, m1=100., m2=1., m3=0.001, G=1.):
        df = pd.read_csv(filepath)
        states = States(df.values, ThreeBody2D(m1, m2, m3, G))
        times_filepath = filepath.replace("u.csv", "t.csv")
        if os.path.exists(times_filepath):
            times = pd.read_csv(times_filepath).values.flatten()
        elif dt is not None:
            times = np.arange(0, len(states)) * dt
        else:
            raise ValueError("Either provide a valid dt or a times file.")
        return cls(times, states)

    @classmethod
    def from_pq(cls, times, pq, m1=100., m2=1., m3=0.001, G=1.):
        threebody = ThreeBody2D(m1, m2, m3, G)
        u = threebody.convert_pq_to_vx(pq)
        return cls(times, States(u, threebody))
    
    @classmethod
    def from_vx(cls, times, vx, m1=100., m2=1., m3=0.001, G=1.):
        threebody = ThreeBody2D(m1, m2, m3, G)
        return cls(times, States(vx, threebody))