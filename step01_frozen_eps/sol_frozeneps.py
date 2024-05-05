"""
frozen approach, to get eps (turb. dissipation rate)
including kep_frozen_eps and kepMK_frozen_eps
"""
import numpy as np
from subfunc.subfuncs import sol_eqn

def kep_frozen_eps(mesh, u, k, e, ReTau):
    n = mesh.nPoints
    r = np.ones(n)
    mu = np.ones(n)/ReTau
    d = np.minimum(mesh.y, mesh.y[-1] - mesh.y)
    # yplus = d * retau
    # Model constants
    cmu = 0.09
    sige = 1.3
    Ce1 = 1.44
    Ce2 = 1.92
    # Model functions
    mut = cmu * r / e * np.power(k, 2)
    mut[1:-1] = np.minimum(np.maximum(mut[1:-1], 1.0e-10), 100.0)

    # Turbulent production: Pk = mut*dudy^2
    Pk = mut * np.power(mesh.ddy @ u, 2)

    # ---------------------------------------------------------------------
    # e-equation
    # effective viscosity
    mueff = mu + mut / sige
    fs = fd = np.ones(n)
    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = np.einsum('i,ij->ij', mueff * fd, mesh.d2dy2) \
        + np.einsum('i,ij->ij', (mesh.ddy @ mueff) * fd, mesh.ddy)
    # Left-hand-side, implicitly treated source term
    np.fill_diagonal(A, A.diagonal() - Ce2 * r * e / k / fs)

    # Right-hand-side
    b = -e[1:-1] / k[1:-1] * Ce1 * Pk[1:-1]
    # Wall boundary conditions
    e[0] = mu[0] / r[0] * k[1] / np.power(d[1], 2)
    e[-1] = mu[-1] / r[-1] * k[-2] / np.power(d[-2], 2)
    # Solve eps equation
    e = sol_eqn(e * fs, A, b, 0.8) / fs
    e[1:-1] = np.maximum(e[1:-1], 1.e-12)
    
    return mut, e

def kepMK_frozen_eps(mesh, u, k, e, retau):
    n = mesh.nPoints
    r = np.ones(n)
    mu = np.ones(n)/retau
    d = np.minimum(mesh.y, mesh.y[-1] - mesh.y)
    yplus = d * retau
    # Model constants
    cmu = 0.09
    sige = 1.3
    Ce1 = 1.4
    Ce2 = 1.8
    # Model functions
    ReTurb = r * np.power(k, 2) / (mu * e)
    f2 = (1 - 2 / 9 * np.exp(-np.power(ReTurb / 6, 2))) * np.power(1 - np.exp(-yplus / 5), 2)
    fmue = (1 - np.exp(-yplus / 70)) * (1.0 + 3.45 / np.power(ReTurb, 0.5))
    fmue[0] = fmue[-1] = 0.0
    # eddy viscosity
    mut = cmu * fmue * r / e * np.power(k, 2)
    mut[1:-1] = np.minimum(np.maximum(mut[1:-1], 1.0e-10), 100.0)

    # Turbulent production: Pk = mut*dudy^2
    Pk = mut * np.power(mesh.ddy @ u, 2)

    # ---------------------------------------------------------------------
    # e-equation
    # effective viscosity
    mueff = mu + mut / sige
    fs = fd = np.ones(n)
    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = np.einsum('i,ij->ij', mueff * fd, mesh.d2dy2) \
        + np.einsum('i,ij->ij', (mesh.ddy @ mueff) * fd, mesh.ddy)
    # Left-hand-side, implicitly treated source term
    np.fill_diagonal(A, A.diagonal() - Ce2 * f2 * r * e / k / fs)

    # Right-hand-side
    b = -e[1:-1] / k[1:-1] * Ce1 * Pk[1:-1]
    # Wall boundary conditions
    e[0] = mu[0] / r[0] * k[1] / np.power(d[1], 2)
    e[-1] = mu[-1] / r[-1] * k[-2] / np.power(d[-2], 2)
    # Solve eps equation
    e = sol_eqn(e * fs, A, b, 0.8) / fs
    e[1:-1] = np.maximum(e[1:-1], 1.e-12)

    return mut, e


def kepLB_frozen_eps(mesh, u, k, e, retau):
    n = mesh.nPoints
    r = np.ones(n)
    mu = np.ones(n)/retau
    d = np.minimum(mesh.y, mesh.y[-1] - mesh.y)
    ystar = np.power(k, 0.5) * d* retau

    # Model constants
    cmu = 0.09
    sige = 1.3
    Ce1 = 1.44
    Ce2 = 1.92
    # Model functions
    ReTurb = r * np.power(k, 2) / (mu * e)
    f2 = (1 - np.exp(-1*np.power(ReTurb , 2))) 
    fmue = np.power((1-np.exp(-0.0165*ystar)),2) * (1 + 20.5/ReTurb)
    fmue[0] = fmue[-1] = 0.0
    # eddy viscosity
    mut = cmu * fmue * r / e * np.power(k, 2)
    mut[1:-1] = np.minimum(np.maximum(mut[1:-1], 1.0e-10), 100.0)

    # Turbulent production: Pk = mut*dudy^2
    Pk = mut * np.power(mesh.ddy @ u, 2)

    # ---------------------------------------------------------------------
    # e-equation
    # effective viscosity
    mueff = mu + mut / sige
    fs = fd = np.ones(n)
    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = np.einsum('i,ij->ij', mueff * fd, mesh.d2dy2) \
        + np.einsum('i,ij->ij', (mesh.ddy @ mueff) * fd, mesh.ddy)
    # Left-hand-side, implicitly treated source term
    np.fill_diagonal(A, A.diagonal() - Ce2 * f2 * r * e / k / fs)

    # Right-hand-side
    b = -e[1:-1] / k[1:-1] * Ce1 * Pk[1:-1]
    # Wall boundary conditions
    e[0] = mu[0] / r[0] * k[1] / np.power(d[1], 2)
    e[-1] = mu[-1] / r[-1] * k[-2] / np.power(d[-2], 2)
    # Solve eps equation
    e = sol_eqn(e * fs, A, b, 0.8) / fs
    e[1:-1] = np.maximum(e[1:-1], 1.e-12)

    return mut, e



def sst_frozen_eps(mesh, u, k, om, retau):
    n = mesh.nPoints
    r = np.ones(n)
    mu = np.ones(n)/retau
    # model constants
    sigma_om1 = 0.5
    sigma_om2 = 0.856
    beta_1 = 0.075
    beta_2 = 0.0828
    betaStar = 0.09
    a1 = 0.31
    alfa_1 = 5.0/9.0 
    alfa_2 = 0.44 
    # Relaxation factors
    underrelaxOm = 0.4

    # required gradients
    dkdy = mesh.ddy @ k
    domdy = mesh.ddy @ om

    wallDist = np.minimum(mesh.y, mesh.y[-1] - mesh.y)
    wallDist = np.maximum(wallDist, 1.0e-8)

    # VortRate = StrainRate in fully developed channel
    strMag = np.absolute(mesh.ddy @ u)

    # Blending functions
    CDkom = 2.0 * sigma_om2 * r / om * dkdy * domdy
    gamma1 = 500.0 * mu / (r * om * wallDist * wallDist)
    gamma2 = 4.0 * sigma_om2 * r * k / (wallDist * wallDist * np.maximum(CDkom, 1.0e-20))
    gamma3 = np.sqrt(k) / (betaStar * om * wallDist)
    gamma = np.minimum(np.maximum(gamma1, gamma3), gamma2)
    bF1 = np.tanh(np.power(gamma, 4.0))
    gamma = np.maximum(2.0 * gamma3, gamma1)
    bF2 = np.tanh(np.power(gamma, 2.0))

    # more model constants
    alfa = alfa_1 * bF1 + (1 - bF1) * alfa_2
    beta = beta_1 * bF1 + (1 - bF1) * beta_2
    sigma_om = sigma_om1 * bF1 + (1 - bF1) * sigma_om2

    # Eddy viscosity
    zeta = np.minimum(1.0 / om, a1 / (strMag * bF2))
    mut = r * k * zeta
    mut = np.minimum(np.maximum(mut, 0.0), 100.0)

    # ---------------------------------------------------------------------
    # om-equation

    # effective viscosity
    mueff = mu + sigma_om * mut
    fs = np.ones(n)

    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = np.einsum('i,ij->ij', mueff, mesh.d2dy2) \
        + np.einsum('i,ij->ij', mesh.ddy @ mueff, mesh.ddy)

    # implicitly treated source term
    np.fill_diagonal(A, A.diagonal() - beta * r * om / fs)

    # Right-hand-side
    b = -alfa[1:-1] * r[1:-1] * strMag[1:-1] * strMag[1:-1] - (1 - bF1[1:-1]) * CDkom[1:-1]

    # Wall boundary conditions
    om[0] = 60.0 * mu[0] / beta_1 / r[0] / wallDist[2] / wallDist[1]
    om[-1] = 60.0 * mu[-1] / beta_1 / r[-1] / wallDist[-2] / wallDist[-2]

    # Solve
    om = sol_eqn(om * fs, A, b, underrelaxOm) / fs
    om[1:-1] = np.maximum(om[1:-1], 1.e-12)

    
    e = betaStar*k*om
    return mut, e