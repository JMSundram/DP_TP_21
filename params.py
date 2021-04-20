import numpy as np
from numba.types import double, int32
from numba.experimental import jitclass

# Initialize parameter class
spec = [('alpha', double), ('T', int32), ('TR', int32), ('TH', int32), ('rho', double),
        ('beta', double), ('sigma_xi', double), ('sigma_psi', double), ('sigma_d', double),
        ('pi', double), ('mu', double), ('R', double), ('delta', double), ('m_max', double),
        ('m_phi', double), ('h_max', double), ('h_phi', double), ('Nxi', int32), ('Npsi', int32),
        ('Nd', int32), ('Nm', int32), ('Nh', int32), ('sim_mini', double), ('simN', int32),
        ('seed', int32), ('G', double[:]), ('psi_vec', double[:]), ('xi_vec', double[:]),
        ('d_vec', double[:]), ('w', double[:]), ('Nshocks', int32), ('grid_h', double[:]),
        ('grid_m', double[:]), ('grid_a', double[:]), ('z_mode', int32), ('Na', int32), ('a_max', double)]

@jitclass(spec)
class par_class:
    def __init__(self):    
        # Scalar parameters
        self.alpha = 0.125 # Fraction of income earned as holiday pay
        self.T = 50 # Number of periods
        self.TR = 35 # Retirement age
        self.TH = 19 # Holiday payout age
        self.rho  = 3.0 # CRRA coeficient
        self.beta = 0.97 # Discount factor
        self.sigma_xi = 0.1 # Standard deviaton of transitory shock
        self.sigma_psi = 0.1 # Standard deviaton of permanent shock
        self.sigma_d = 0.1 # Standard deviation on holiday pay growth
        self.pi = 0.075 # Probability of low income shock
        self.mu = 0.2 # Value of low income shock
        self.R = 1.03 # Asset return factor
        self.delta = 1.05 # Holiday pay return factor
        self.m_max = 10.0 # Maximum point in grid for m
        self.m_phi = 1.1 # Curvature of grid for m
        self.h_max = 1.0 # Maximum point in grid for h
        self.h_phi = 1.1 # Curvature of grid for h
        self.Nxi = 6 # Number of quadrature points for xi
        self.Npsi = 6 # Number of quadrature points for psi
        self.Nd = 6 # Number of quadrature points for d
        self.Nm = 100 # Number of points in grid for m
        self.Na = 100 # Number of points in grid for a
        self.a_max = 10.0 # Maximum point in grid for a
        self.Nh = 20 # Number of points in grid for h
        self.z_mode = 2 # Mode for z
        self.sim_mini = 2.5 # Initial m in simulation
        self.simN = 100_000 # Number of persons in simulation
        self.seed = 2021 # Seed for random draws
        
        # Life cycle
        self.G = np.ones(self.T)
        self.G[:self.TR-1] = np.linspace(1.1, 0.99, self.TR-1)
        self.G[self.TR-1] = 0.9
