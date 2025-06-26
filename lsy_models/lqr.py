import numpy as np
import scipy.linalg


class LQRController:
    """Linear Quadratic Regulator controller for drone trajectory tracking."""
    
    def __init__(self, A, B, Q, R, dt):
        """
        Initialize LQR controller.
        
        Args:
            A: Continuous-time system matrix
            B: Continuous-time input matrix  
            Q: State penalty matrix
            R: Control penalty matrix
            dt: Sampling time
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.dt = dt
        
        # Discretize system
        self.Ad, self.Bd = self._c2d(A, B, dt)
        
        # Solve DARE and compute feedback gain
        self.P = self._solve_dare()
        self.K = self._compute_gain()
        
    def _c2d(self, A, B, dt):
        """Convert continuous-time system to discrete-time."""
        n = A.shape[0]
        m = B.shape[1]
        
        # Construct augmented matrix
        M = np.block([
            [A, B],
            [np.zeros((m, n)), np.zeros((m, m))]
        ])
        
        # Matrix exponential
        Mexp = scipy.linalg.expm(M * dt)
        
        # Extract discrete matrices
        Ad = Mexp[:n, :n]
        Bd = Mexp[:n, n:]
        
        return Ad, Bd
    
    def _solve_dare(self):
        """Solve Discrete Algebraic Riccati Equation."""
        return scipy.linalg.solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)
    
    def _compute_gain(self):
        """Compute LQR feedback gain matrix."""
        return np.linalg.inv(self.Bd.T @ self.P @ self.Bd + self.R) @ (self.Bd.T @ self.P @ self.Ad)
    
    def compute_control(self, x_current, x_ref, u_eq):
        """
        Compute control input using LQR feedback law.
        
        Args:
            x_current: Current state
            x_ref: Reference state
            u_eq: Equilibrium control input
            
        Returns:
            Control input
        """
        return u_eq - self.K @ (x_current - x_ref)
    
    def get_gain_matrix(self):
        """Return the LQR gain matrix."""
        return self.K
    
    def get_cost_matrix(self):
        """Return the solution to the DARE."""
        return self.P


def design_lqr_controller(A, B, Q, R, dt):
    """
    Factory function to create and return an LQR controller.
    
    Args:
        A: Continuous-time system matrix
        B: Continuous-time input matrix
        Q: State penalty matrix
        R: Control penalty matrix
        dt: Sampling time
        
    Returns:
        LQRController instance
    """
    return LQRController(A, B, Q, R, dt)
