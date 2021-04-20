import numpy as np
import tools
from numba import njit, prange
from numba.experimental import jitclass
from numba.types import double, int32

def run_model(par):
    # 1. Prepare grids and allocate solution
    # Gauss Hermite
    psi, psi_w = tools.GaussHermite_lognorm(sigma=par.sigma_psi,n=par.Npsi)
    xi, xi_w = tools.GaussHermite_lognorm(sigma=par.sigma_xi,n=par.Nxi)
    d, d_w = tools.GaussHermite_lognorm(sigma=par.sigma_d, n=par.Nd)
    
    # Add low income shock to xi
    if par.pi > 0:
        # Weights
        xi_w *= (1.0-par.pi)
        xi_w = np.insert(xi_w,0,par.pi)

        # Values
        xi = (xi-par.mu*par.pi)/(1.0-par.pi)
        xi = np.insert(xi,0,par.mu)

    # Vectorize tensor product of shocks and total weight
    psi_vec,xi_vec,d_vec = np.meshgrid(psi,xi,d,indexing='ij')
    psi_w_vec,xi_w_vec,d_w_vec = np.meshgrid(psi_w,xi_w,d_w,indexing='ij')

    par.psi_vec = psi_vec.ravel()
    par.xi_vec = xi_vec.ravel()
    par.d_vec = d_vec.ravel()
    par.w = xi_w_vec.ravel()*psi_w_vec.ravel()*d_w_vec.ravel()

    assert 1-np.sum(par.w) < 1e-8 # Check if weights sum to 1

    # Count number of shock nodes
    par.Nshocks = par.w.size

    # Create grids
    par.grid_m = tools.nonlinspace(1e-6,par.m_max,par.Nm,par.m_phi)
    par.grid_h = tools.nonlinspace(0,par.h_max,par.Nh,par.h_phi)
    par.grid_a = tools.nonlinspace(1e-6,par.m_max,par.Nm,par.m_phi)
     
    # Create solution and simulation
    spec = [('m', double[:,:,:,:]), ('c', double[:,:,:,:]), ('inv_v', double[:,:,:,:])]
    @jitclass(spec)
    class sol_:
        def __init__(self): pass
        
    spec = [('m', double[:,:]), ('c', double[:,:]), ('a', double[:,:]), ('p', double[:,:]), ('y', double[:,:]),
            ('psi', double[:,:]), ('xi', double[:,:]), ('d', double[:,:]), ('P', double[:,:]), ('Y', double[:,:]),
            ('M', double[:,:]), ('C', double[:,:]), ('A', double[:,:]), ('z', int32[:]), ('h', double[:,:]),
            ('H', double[:,:])]
    @jitclass(spec)
    class sim_:
        def __init__(self): pass

    # Allocate memory for solution
    sol = sol_()
    sol_shape = (par.T,par.Na+1,2,par.Nh)
    sol.m = np.zeros(sol_shape)
    sol.c = np.zeros(sol_shape)
    sol.inv_v = np.zeros((par.T,par.Nm,2,par.Nh))
    
    # Allocate memory for simulation
    sim = sim_()
    sim_shape = (par.simN,par.T)
    sim.m = np.zeros(sim_shape)
    sim.c = np.zeros(sim_shape)
    sim.a = np.zeros(sim_shape)
    sim.p = np.zeros(sim_shape)
    sim.y = np.zeros(sim_shape)
    sim.psi = np.zeros(sim_shape)
    sim.xi = np.zeros(sim_shape)
    sim.d = np.zeros(sim_shape)
    sim.P = np.zeros(sim_shape)
    sim.Y = np.zeros(sim_shape)
    sim.M = np.zeros(sim_shape)
    sim.C = np.zeros(sim_shape)
    sim.A = np.zeros(sim_shape)
    sim.z = np.zeros((par.simN), dtype=int)
    sim.h = np.zeros(sim_shape)
    sim.H = np.zeros(sim_shape)
    
    # 2. Solve model by EGM
    @njit
    def utility(c, rho):
        return c**(1-rho)/(1-rho)    
    
    @njit
    def marg_utility(c, rho):
        return c**(-rho)      
    
    @njit
    def inv_marg_utility(u, rho):
        return u**(-1/rho)  
    
    @njit
    def value_of_choice(c,t,m,h,z,par,sol):
        # End-of-period assets
        a = m-c
        
        # Calculate inverse value-of-choice in next period
        still_working_next_period = t+1 <= par.TR-1
        if still_working_next_period:
            fac_vec = par.G[t]*par.psi_vec
            w = par.w
            xi = par.xi_vec
            
            if t+1 < par.TH-1:
                h_plus_vec = np.repeat(0.0, len(fac_vec))
            elif t+1 == par.TH-1:
                #h_plus_vec = np.repeat(0.0, len(fac))
                h_plus_vec = np.repeat(par.alpha, len(fac_vec))
            elif t+1 > par.TH-1 and z == 0:
                h_plus_vec = np.repeat(((par.d_vec+par.delta-1)/fac_vec)*h, len(fac_vec))
            elif t+1 > par.TH-1 and z == 1:
                h_plus_vec = np.repeat(0.0, len(fac_vec))
            
            if t+1 == par.TH and z == 1:
                h_term_vec = ((par.d_vec+par.delta-1)/fac_vec)*h
            else:
                h_term_vec = np.repeat(0.0,len(fac_vec))
                
            m_plus_vec = (par.R/fac_vec)*a + xi + h_term_vec
            inv_v_plus_vec = tools.interp_2d_vec(par.grid_m, par.grid_h, sol.inv_v[t+1,:,z,:], m_plus_vec, h_plus_vec)
        else:
            fac = par.G[t]

            if t+1 == par.TR and z == 0:
                h_term = (par.delta/fac)*h
            else:
                h_term = 0.0
            
            m_plus = (par.R/fac)*a + 1 + h_term
            inv_v_plus = tools.interp_2d(par.grid_m, par.grid_h, sol.inv_v[t+1,:,z,:], m_plus, 0.0)

        # Value-of-choice
        if still_working_next_period:
            v_plus_vec = 1/inv_v_plus_vec
            total = utility(c,par.rho) + par.beta*np.sum(w*fac_vec**(1-par.rho)*v_plus_vec)
        else:
            v_plus = 1/inv_v_plus
            total = utility(c,par.rho) + par.beta*fac**(1-par.rho)*v_plus
        return -total

    # Last period (= consume all)
    @njit   
    def solve_egm(par,sol,sim):
        for z in [0, 1]:
            for i_h in range(par.Nh):
                sol.m[-1,:,z,i_h] = np.linspace(0,par.a_max,par.Na+1)
                sol.c[-1,:,z,i_h] = sol.m[-1,:,z,i_h]
                
            # Before last period
            for t in range(par.T-2,-1,-1):
                for i_h, h in enumerate(par.grid_h):
                    if t >= par.TR and i_h != 0: # After retirement, solution is independent of z and h
                        sol.c[t,:,z,i_h] = sol.c[t,:,0,0]
                        sol.m[t,:,z,i_h] = sol.m[t,:,0,0]
                    elif t >= par.TH and z == 1 and i_h != 0: # After early holiday pay, solution is independent of h
                        sol.c[t,:,z,i_h] = sol.c[t,:,z,0]
                        sol.m[t,:,z,i_h] = sol.m[t,:,z,0]
                    elif t < par.TH-1 and i_h != 0: # Before holiday pay decision, solution is independent of z and h
                        sol.c[t,:,z,i_h] = sol.c[t,:,0,0]
                        sol.m[t,:,z,i_h] = sol.m[t,:,0,0]
                    else:
                        # Loop over end-of-period assets
                        for i_a in range(1,par.Na+1):
                            a = par.grid_a[i_a-1]
                            still_working_next_period = t+1 <= par.TR-1
                            if still_working_next_period:
                                fac_vec = par.G[t]*par.psi_vec
                                w = par.w
                                xi = par.xi_vec
                                
                                if t+1 < par.TH-1:
                                    h_plus_vec = np.repeat(0.0, len(fac_vec))
                                elif t+1 == par.TH-1:
                                    h_plus_vec = np.repeat(par.alpha, len(fac_vec))
                                elif t+1 > par.TH-1 and z == 0:
                                    h_plus_vec = np.repeat(((par.d_vec+par.delta-1)/fac_vec)*h, len(fac_vec))
                                elif t+1 > par.TH-1 and z == 1:
                                    h_plus_vec = np.repeat(0.0, len(fac_vec))
                                
                                if t+1 == par.TH and z == 1:
                                    h_term_vec = ((par.d_vec+par.delta-1)/fac_vec)*h
                                else:
                                    h_term_vec = np.repeat(0.0,len(fac_vec))
                                    
                                m_plus_vec = (par.R/fac_vec)*a + xi + h_term_vec
                                
                                # Future c
                                c_plus_vec = tools.interp_2d_vec(sol.m[t+1,:,z,i_h], par.grid_h, sol.c[t+1,:,z,:], m_plus_vec, h_plus_vec)
                                
                                # Average future marginal utility
                                marg_u_plus_vec = marg_utility(fac_vec*c_plus_vec,par.rho)
                                avg_marg_u_plus_vec = np.sum(w*marg_u_plus_vec)
                                
                                # Current c
                                sol.c[t,i_a,z,i_h] = inv_marg_utility(par.beta*par.R*avg_marg_u_plus_vec,par.rho)
                                
                                # Current m
                                sol.m[t,i_a,z,i_h] = a + sol.c[t,i_a,z,i_h]
                            else:
                                fac = par.G[t]
        
                                if t+1 == par.TR and z == 0:
                                    h_term = (par.delta/fac)*h
                                else:
                                    h_term = 0.0
                                
                                m_plus = (par.R/fac)*a + 1 + h_term
                                
                                # Future c
                                c_plus = tools.interp_2d(sol.m[t+1,:,z,i_h], par.grid_h, sol.c[t+1,:,z,:], m_plus, 0.0)
                                
                                # Average future marginal utility
                                marg_u_plus = marg_utility(fac*c_plus,par.rho)
                                avg_marg_u_plus = marg_u_plus
                                
                                # Current c
                                sol.c[t,i_a,z,i_h] = inv_marg_utility(par.beta*par.R*avg_marg_u_plus,par.rho)
                                
                                # Current m
                                sol.m[t,i_a,z,i_h] = a + sol.c[t,i_a,z,i_h]
        
                        # Add zero consumption
                        sol.m[t,0,z,i_h] = 0.0
                        sol.c[t,0,z,i_h] = 0.0
                
    @njit(parallel=True)
    def value_functions(par,sol,sim):
        # Compute value function
        for z in [0, 1]:
            for i_h in prange(par.Nh):
                for i,c in enumerate(par.grid_m):
                    sol.inv_v[-1,i,z,i_h] = 1.0/utility(c,par.rho)
    
            # Before last period
            for t in range(par.T-2,-1,-1):
                for i_h in range(par.Nh): #prange
                    h = par.grid_h[i_h]
                    if t >= par.TR and i_h != 0: # After retirement, solution is independent of z and h
                        sol.inv_v[t,:,z,i_h] = sol.inv_v[t,:,0,0]
                    elif t >= par.TH and z == 1 and i_h != 0: # After early holiday pay, solution is independent of h
                        sol.inv_v[t,:,z,i_h] = sol.inv_v[t,:,z,0]
                    elif t < par.TH-1 and i_h != 0: # Before holiday pay decision, solution is indenpendent of z and h
                        sol.inv_v[t,:,z,i_h] = sol.inv_v[t,:,0,0]
                    else:
                        for i_m in prange(par.Nm):
                            m = par.grid_m[i_m]
                            c = tools.interp_linear_1d_scalar(sol.m[t,:,z,i_h], sol.c[t,:,z,i_h], m)
                            v = value_of_choice(c,t,m,h,z,par,sol)
                            sol.inv_v[t,i_m,z,i_h] = -1.0/v
                            
    # Solve by EGM
    solve_egm(par,sol,sim)
    value_functions(par,sol,sim)

    # 3. Simulate model
        
    # Set seed
    np.random.seed(par.seed)

    # Shocks
    _shocki = np.random.choice(par.Nshocks,size=(par.simN,par.T),p=par.w)
    sim.psi[:] = par.psi_vec[_shocki]
    sim.xi[:] = par.xi_vec[_shocki]
    sim.d[:] = par.d_vec[_shocki]

    # Initial values
    sim.m[:,0] = par.sim_mini 
    sim.p[:,0] = 0.0

    @njit
    def nearest(points,target,norm_fact):
        target[0] = target[0]/norm_fact
        dist = np.sum((points - target)**2, axis=1)
        return np.argmin(dist)
    
    @njit(parallel=True)
    def simulate_time_loop(par,sol,sim):
        # Prepare irregular interpolation
        num_points = par.Nh*(par.Nm+1)
        points = np.zeros((par.T,2,num_points,2))
        values_c = np.zeros((par.T,2,num_points))
        for t in range(par.T):
            for z in [0,1]:
                for i_h,h_loop in enumerate(par.grid_h):
                    m_grid = sol.m[t,:,z,i_h]
                    for i_m,m_loop in enumerate(m_grid):
                        values_c[t,z,i_h*(par.Nm+1)+i_m] = sol.c[t,i_m,z,i_h]
                    points[t,z,(par.Nm+1)*i_h:(par.Nm+1)*(i_h+1),:] = np.array(list(zip(m_grid, [h_loop]*len(m_grid))))
        
        # Normalize points
        norm_fact = par.m_max/par.h_max
        points[:,:,:,0] = points[:,:,:,0]/norm_fact

        # Unpack (helps numba)
        m = sim.m
        p = sim.p
        y = sim.y
        c = sim.c
        a = sim.a
        h = sim.h
        z = sim.z
        
        # loop over first households and then time
        for i in prange(par.simN):
            for t in range(par.T):
                # Consumption
                #c[i,t] = griddata(points[t,z[i],:,:], values_c[t,z[i],:], (m[i,t], h[i,t]), method='nearest')*1.0
                nearest_row = nearest(points[t,z[i],:,:], np.array([m[i,t], h[i,t]]), norm_fact)
                c[i,t] = values_c[t,z[i],nearest_row]
                a[i,t] = m[i,t] - c[i,t]

                if t < par.T-1:
                    still_working_next_period = t+1 <= par.TR-1
                    if still_working_next_period:
                        fac = par.G[t]*sim.psi[i,t+1]
                        xi = sim.xi[i,t+1]

                        if t+1 < par.TH-1:
                            h[i,t+1] = 0.0
                        elif t+1 == par.TH-1:
                            h[i,t+1] = par.alpha
                        elif t+1 > par.TH-1 and z[i] == 0:
                            h[i,t+1] = ((sim.d[i,t+1]+par.delta-1)/fac)*h[i,t]
                        elif t+1 > par.TH-1 and z[i] == 1:
                            h[i,t+1] = 0.0

                        if t+1 == par.TH and z[i] == 1:
                            h_term = ((sim.d[i,t+1]+par.delta-1)/fac)*h[i,t]
                        else:
                            h_term = 0.0

                        m[i,t+1] = (par.R/fac)*a[i,t] + xi + h_term
                        p[i,t+1] = np.log(par.G[t]) + p[i,t] + np.log(sim.psi[i,t+1])   
                        if sim.xi[i,t+1] > 0:
                            y[i,t+1] = p[i,t+1] + np.log(sim.xi[i,t+1])
                        else:
                            y[i,t+1] = -np.inf
                    else:
                        fac = par.G[t]
                        xi = 1

                        h[i,t+1] = 0.0

                        if t+1 == par.TR and z[i] == 0:
                            h_term = (par.delta/fac)*h[i,t]
                        else:
                            h_term = 0.0

                        m[i,t+1] = (par.R/fac)*a[i,t] + xi + h_term
                        p[i,t+1] = np.log(par.G[t]) + p[i,t]
                        y[i,t+1] = p[i,t+1]

                    if t+1 == par.TH-1:
                        if par.z_mode == 2:
                            inv_v0 = tools.interp_2d(par.grid_m, par.grid_h, sol.inv_v[t+1,:,0,:], m[i,t+1], h[i,t+1])
                            inv_v1 = tools.interp_2d(par.grid_m, par.grid_h, sol.inv_v[t+1,:,1,:], m[i,t+1], h[i,t+1])
                            if inv_v0 < inv_v1:
                                z[i] = 0
                            else:
                                z[i] = 1
                        elif par.z_mode == 1:
                            z[i] = 1
                        elif par.z_mode == 0:
                            z[i] = 0
                            
    # Simulate model                     
    simulate_time_loop(par,sol,sim)

    # Renormalize
    sim.P[:,:] = np.exp(sim.p)
    sim.Y[:,:] = np.exp(sim.y)
    sim.M[:,:] = sim.m*sim.P
    sim.C[:,:] = sim.c*sim.P
    sim.A[:,:] = sim.a*sim.P
    sim.H[:,:] = sim.h*sim.P

    return sol, sim