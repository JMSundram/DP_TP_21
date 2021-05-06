import numpy as np
import tools
from numba import njit

@njit
def calculate_euler_errors(sol,sim,par,grid_m):
    def marg_utility(c, rho):
            return c**(-rho)  

    euler_residual = np.nan + np.zeros((par.simN,par.T-1)) 
    a = np.nan + np.zeros((par.simN,par.T-1))

    for t in range(par.T-1):
        for i in range(par.simN):
            # Initialize
            m = sim.m[i,t]
            c = sim.c[i,t]
            a[i,t] = m-c
            h = sim.h[i,t]
            if t < par.TH-1:
                z = 0
            else:
                z = sim.z[i]

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

                m_plus_vec = (par.R/fac_vec)*a[i,t] + xi + h_term_vec

                # Future c
                c_plus_vec = tools.interp_2d_vec(grid_m, par.grid_h, sol.c[t+1,:,z,:], m_plus_vec, h_plus_vec)

                # Average future marginal utility
                marg_u_plus_vec = marg_utility(fac_vec*c_plus_vec,par.rho)
                avg_marg_u_plus_vec = np.sum(w*marg_u_plus_vec)

                euler_residual[i,t] = marg_utility(c,par.rho)-par.beta*par.R*avg_marg_u_plus_vec
            else:
                fac = par.G[t]

                if t+1 == par.TR and z == 0:
                    h_term = (par.delta/fac)*h
                else:
                    h_term = 0.0

                m_plus = (par.R/fac)*a[i,t] + 1 + h_term

                # Future c
                c_plus = tools.interp_2d(grid_m, par.grid_h, sol.c[t+1,:,z,:], m_plus, 0.0)

                # Average future marginal utility
                marg_u_plus = marg_utility(fac*c_plus,par.rho)
                avg_marg_u_plus = marg_u_plus

                euler_residual[i,t] = marg_utility(c,par.rho)-par.beta*par.R*avg_marg_u_plus

    # 4. Calculate the average absolute euler residual
    I = (a>0)   # Define an indicator for a bigger than 0 (m>c)
    eulers_indexed = euler_residual.flatten()[I.flatten()]
    euler_error = np.mean(np.abs(eulers_indexed))

    c = (sim.c[:,0:par.T-1])  # The euler error is not defined in last period

    nom_euler_error = np.log10(np.abs(eulers_indexed)/(c.flatten()[I.flatten()]))   
    nom_euler_error = np.mean(nom_euler_error)

    return euler_error, nom_euler_error