import numpy as np 
import pandas as pd


class NCO:
    def __init__(self, epsilon=0.01):
        self.epsilon = epsilon
    
    def compute_kinetic_energy(self, u):
        v1, v2 = u[..., 0], u[..., 1]
        return 0.5 * v1**2 + 0.5 * v2**2 / self.epsilon
    
    def compute_potential_energy(self, u):
        x1, x2 = u[..., 2], u[..., 3]
        return 0.5 * x1**2 + 0.5 * x2**2 * self.epsilon + self.epsilon * x1 * x2 * np.sin(2*(x1 + x2))
    
    def compute_hamiltonian(self, u):
        return self.compute_kinetic_energy(u) + self.compute_potential_energy(u)

    def compute_energies(self, u):
        v1, v2, x1, x2 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
        K1 = 0.5 * v1**2
        K2 = 0.5 * v2**2 / self.epsilon
        V1 = 0.5 * x1**2
        V2 = 0.5 * x2**2 * self.epsilon
        V12 = self.epsilon * x1 * x2 * np.sin(2*(x1 + x2))
        return {
            "H": K1+K2+V1+V2+V12, 
            "K1+V1": K1+V1, 
            "K2+V2": K2+V2, 
            "K2+V2+V12": K2+V2+V12,
            # "K1": K1, 
            # "K2": K2, 
            # "V1": V1, 
            # "V2": V2, 
            "V12": V12
        }

    def compute_errors(self, u, ref_u):
        # Rescale the data for traj error computation: p2 = v2 * m2 = v2 / epsilon
        pq = u * np.array([1., 1/self.epsilon, 1., 1.])
        ref_pq = ref_u * np.array([1., 1/self.epsilon, 1., 1.])

        # Compute trajectory errors
        diff_squares = (pq - ref_pq)**2
        abs_traj_err = np.sqrt(np.sum(diff_squares, axis=-1))
        rel_traj_err = abs_traj_err / np.linalg.norm(ref_pq, axis=-1)
        osc1_abs_traj_err = np.sqrt(np.sum(diff_squares[..., (0, 2)], axis=-1))
        osc1_rel_traj_err = osc1_abs_traj_err / np.linalg.norm(ref_pq[..., (0, 2)], axis=-1)
        osc2_abs_traj_err = np.sqrt(np.sum(diff_squares[..., (1, 3)], axis=-1))
        osc2_rel_traj_err = osc2_abs_traj_err / np.linalg.norm(ref_pq[..., (1, 3)], axis=-1)

        # Compute Hamiltonian errors 
        H = self.compute_hamiltonian(u)
        ref_H = self.compute_hamiltonian(ref_u)
        abs_H_err = np.abs(H - ref_H)
        rel_H_err = abs_H_err / np.abs(ref_H)
        
        return {
            "abs_traj_err": abs_traj_err, 
            "rel_traj_err": rel_traj_err,
            "osc1_abs_traj_err": osc1_abs_traj_err, 
            "osc1_rel_traj_err": osc1_rel_traj_err,
            "osc2_abs_traj_err": osc2_abs_traj_err, 
            "osc2_rel_traj_err": osc2_rel_traj_err,
            "abs_H_err": abs_H_err, 
            "rel_H_err": rel_H_err
        }
    
    def compute_grad_H(self, u):
        """
        dH/dp1 = p1
        dH/dp2 = epsilon * p2 
        dH/dq1 = q1 + epsilon * dU/dq1
        dH/dq2 = epsilon * q2 + epsilon * dU/dq2 

        U(q1, q2) = q1 * q2 * sin(2*q1+2*q2)
        dU/dq1 = q2 * sin(2*q1+2*q2) + q1 * q2 * 2 * cos(2*q1+2*q2)
        dU/dq2 = q1 * sin(2*q1+2*q2) + q1 * q2 * 2 * cos(2*q1+2*q2)
        """
        v1, v2, x1, x2 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
        p1, p2, q1, q2 = v1, v2/self.epsilon, x1, x2

        dUdq1 = q2 * np.sin(2*q1+2*q2) + q1 * q2 * 2 * np.cos(2*q1+2*q2)
        dUdq2 = q1 * np.sin(2*q1+2*q2) + q1 * q2 * 2 * np.cos(2*q1+2*q2)

        dHdp1 = p1
        dHdp2 = self.epsilon * p2
        dHdq1 = q1 + self.epsilon * dUdq1
        dHdq2 = self.epsilon * q2 + self.epsilon * dUdq2

        return np.stack([dHdp1, dHdp2, dHdq1, dHdq2], axis=-1)
    
    def compute_hessian_H(self, u):  ## TODO
        """
        [[d^2H/dp1^2, d^2H/dp1dp2, d^2H/dp1dq1, d^2H/dp1dq2],
         [d^2H/dp2dp1, d^2H/dp2^2, d^2H/dp2dq1, d^2H/dp2dq2],
         [d^2H/dq1dp1, d^2H/dq1dp2, d^2H/dq1^2, d^2H/dq1dq2],
         [d^2H/dq2dp1, d^2H/dq2dp2, d^2H/dq2dq1, d^2H/dq2^2]]
        
        d2H/dp1dp2 = d2H/dp1dq1 = d2H/dp1dq2 = 0
        d2H/dp2dp1 = d2H/dp2dq1 = d2H/dp2dq2 = 0
        d2H/dq1dp1 = d2H/dq1dp2 = 0 
        d2H/dq2dp1 = d2H/dq2dp2 = 0
        d2H/dp1dp1 = 1
        d2H/dp2dp2 = epsilon
        d2H/dq1dq1 = 1 + epsilon * d2U/dq1dq1
        d2H/dq2dq2 = epsilon + epsilon * d2U/dq2dq2
        d2H/dq1dq2 = epsilon * d2U/dq1dq2
        d2H/dq2dq1 = epsilon * d2U/dq2dq1

        U(q1, q2) = q1 * q2 * sin(2*q1+2*q2)
        d2U/dq1dq1 = q2 * 4 * cos(2*q1+2*q2) - q1 * q2 * 4 * sin(2*q1+2*q2)
        d2U/dq2dq2 = q1 * 4 * cos(2*q1+2*q2) - q1 * q2 * 4 * sin(2*q1+2*q2)
        d2U/dq1dq2 = d2U/dq2dq1 = sin(2*q1+2*q2) + (q1+q2) * 2 * cos(2*q1+2*q2) - q1 * q2 * 4 * sin(2*q1+2*q2)
        """
        v1, v2, x1, x2 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
        p1, p2, q1, q2 = v1, v2/self.epsilon, x1, x2

        S = np.sin(2*q1+2*q2)
        C = np.cos(2*q1+2*q2)
        d2Udq1dq1 = q2 * 4 * C - q1 * q2 * 4 * S
        d2Udq2dq2 = q1 * 4 * C - q1 * q2 * 4 * S
        d2Udq1dq2 = S + (q1+q2) * 2 * C - q1 * q2 * 4 * S
        
        d2Hdp1dp1 = np.ones_like(p1)
        d2Hdp2dp2 = np.ones_like(p2) * self.epsilon
        d2Hdq1dq1 = np.ones_like(q1) + self.epsilon * d2Udq1dq1
        d2Hdq2dq2 = np.ones_like(q2) * self.epsilon + self.epsilon * d2Udq2dq2
        d2Hdq1dq2 = d2Hdq2dq1 = self.epsilon * d2Udq1dq2

        hessian = np.zeros((len(u), 4, 4))
        hessian[:, 0, 0] = d2Hdp1dp1
        hessian[:, 1, 1] = d2Hdp2dp2
        hessian[:, 2, 2] = d2Hdq1dq1
        hessian[:, 3, 3] = d2Hdq2dq2
        hessian[:, 2, 3] = d2Hdq1dq2
        hessian[:, 3, 2] = d2Hdq2dq1

        return hessian
    
# def compute_traj_error_elementwise(sol, ref_sol, epsilon=0.01, idx=0):
#     # Rescale the data: p2 = v2 * m2 = v2 / epsilon
#     rescaled_sol = sol * np.array([1., 1/epsilon, 1., 1.])
#     rescaled_ref_sol = ref_sol * np.array([1., 1/epsilon, 1., 1.])
#     abs_err = np.abs(rescaled_sol[..., idx] - rescaled_ref_sol[..., idx])
#     rel_err = abs_err / np.abs(rescaled_ref_sol[..., idx])
#     return abs_err, rel_err


def is_multiple(x, y):
    quotient = x / y
    rounded_quotient = round(quotient)
    return abs(rounded_quotient * y - x) < 1e-9  # Adjust threshold as needed

def find_quotient(x, y):
    quotient = x / y
    rounded_quotient = round(quotient)
    return rounded_quotient


class Trajectory:
    def __init__(self, t, u, epsilon=0.01):
        assert len(t) > 0, "Empty time array"
        assert len(t) == len(u), f"Length mismatch: len(t) = {len(t)}, len(u) = {len(u)}"
        assert u.shape[1] == 4, f"Invalid shape: {u.shape}"
        self.uu = u
        self.tt = t
        self.dt = t[1] - t[0]

        self.epsilon = epsilon
        self.p1 = u[:, 0]
        self.p2 = u[:, 1] / epsilon
        self.q1 = u[:, 2]
        self.q2 = u[:, 3]

    def __repr__(self):
        return f"Trajectory(t_range=[{self.tt[0]}, {self.tt[-1]}], dt={self.dt}, length={len(self.tt)}, u0={self.uu[0]})"
    
    def __len__(self):
        return len(self.tt)
    
    def initial_state(self):
        return self.tt[0], self.uu[0]
    
    def select_between(self, t0, t1):
        idx = (self.tt >= t0) & (self.tt <= t1)
        return Trajectory(self.tt[idx], self.uu[idx])
    
    def select_with_interval(self, Dt):
        assert is_multiple(Dt, self.dt), f"Invalid interval Dt = {Dt}, dt = {self.dt}"
        idx = np.arange(0, len(self.tt), find_quotient(Dt, self.dt))
        return Trajectory(self.tt[idx], self.uu[idx])

    def compare(self, ref_traj):
        matched_traj, matched_ref_traj = Trajectory.intersect(self, ref_traj)
        if matched_traj is None or matched_ref_traj is None:
            print("Trajectories could not be intersected with matched intervals.")
            return None
        print(f"After alignment: \n\ttraj = {matched_traj}")
        errors = NCO(self.epsilon).compute_errors(matched_traj.uu, matched_ref_traj.uu)
        return matched_traj.tt, errors
    
    @classmethod
    def load_from_file(cls, filepath, dt=0.05):
        df = pd.read_csv(filepath)
        u = df.values
        t = np.arange(0, len(u)) * dt
        return cls(t, u)

    @staticmethod
    def intersect(traj1, traj2, match_dt=True):
        t0 = max(traj1.tt[0], traj2.tt[0])
        t1 = min(traj1.tt[-1], traj2.tt[-1])
        if match_dt:
            dt = max(traj1.dt, traj2.dt)
            try: 
                return traj1.select_between(t0, t1).select_with_interval(dt), traj2.select_between(t0, t1).select_with_interval(dt)
            except ValueError as e:
                print(e)
                return None, None
        else:
            return traj1.select_between(t0, t1), traj2.select_between(t0, t1)
