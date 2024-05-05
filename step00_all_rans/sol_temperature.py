# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 10:28:34 2019

@author: xiaoweix2
"""
import numpy as np
from subfunc.subfuncs import sol_eqn

def sol_temper(mesh, temperature, r,mut, Pr, ReTau):
    """
    solve Temperature equation, isothermal bc, with force, natural, and mixed conveciton cases
    """
    n = mesh.nPoints
    # molecular thermal conductivity:
    lam = np.ones(n) / (ReTau * Pr)
    # turbulent prandtl: assume = 0.9
    turbpr = 0.9*np.ones(n)
    # diffusion matrix: lamEff*d2phi/dy2 + dlamEff/dy dphi/dy
    a = np.einsum('i,ij->ij', mesh.ddy @ (lam + mut / turbpr), mesh.ddy) + np.einsum('i,ij->ij', lam + mut / turbpr,mesh.d2dy2) 
                                                                                                                                                            
    # Isothermal BC
    temperature[0] = 0.5
    temperature[-1] = -0.5

    b =  np.zeros(n - 2)  # b
    # Solve
    temperature = sol_eqn(temperature, a, b, 0.95)
    
    r = np.ones(n)
    mu = np.ones(n)/ ReTau
    return r, mu, temperature