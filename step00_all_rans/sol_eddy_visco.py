"""
In turbulence_models.py several models, including SA, k-epsilon, k-omega SST, 
v2-f, and cess are implemented for RANS simulation
"""
import numpy as np
from subfunc.subfuncs import sol_eqn

def kepXu(mesh, u, k, e, vv, r, mu, ReTau):
    n = mesh.nPoints
    d = np.minimum(mesh.y, mesh.y[-1] - mesh.y) #wall distance, y_n
#    yplus = d * retau
    ystar = d * np.sqrt(k)/mu

    # Model constants
#    cmu = 0.09
    sigk = 1.0
    sige = 1.3
    Ce1 = 1.44
    Ce2 = 1.92
    ## vv
    vv =k *((7.19e-3)*ystar-(4.33e-5)*np.power(ystar,2.0)+(8.8e-8)*np.power(ystar,3.0))
    ystar_vv=np.sqrt(vv)*d/mu   
    # Model functions
    lmu=0.544*d/(1+5.025e-4*pow(ystar_vv,1.65))
    leps=8.8*d/(1+10/ystar_vv +5.15e-2*ystar_vv)  
    
    mut =np.sqrt(vv)*lmu
    e=np.sqrt(vv)*k/leps

    # Turbulent production: Pk = mut*dudy^2
    Pk = mut * np.power(mesh.ddy @ u, 2)
    # ---------------------------------------------------------------------
    # e-equation
    mueff = mu + mut / sige
    fs = fd = f2 = np.ones(n)    
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
    # ---------------------------------------------------------------------
    # k-equation
    # effective viscosity
    mueff = mu + mut / sigk
    fs = fd = np.ones(n)
    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = np.einsum('i,ij->ij', mueff * fd, mesh.d2dy2) \
        + np.einsum('i,ij->ij', (mesh.ddy @ mueff) * fd, mesh.ddy)
    # implicitly treated source term
    np.fill_diagonal(A, A.diagonal() - r * e / k / fs)
    # Right-hand-side
    b = -Pk[1:-1]

    # Wall boundary conditions
    k[0] = k[-1] = 0.0
    # Solve TKE
    k = sol_eqn(k * fs, A, b, 0.7) / fs
    k[1:-1] = np.maximum(k[1:-1], 1.e-12)

    return mut, k, e, Pk


def kep(mesh, u, k, e, r, mu):
    n = mesh.nPoints
    d = np.minimum(mesh.y, mesh.y[-1] - mesh.y)
    # yplus = d * retau
    # Model constants
    cmu = 0.09
    sigk = 1.0
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

    # ---------------------------------------------------------------------
    # k-equation
    # effective viscosity
    mueff = mu + mut / sigk
    fs = fd = np.ones(n)
    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = np.einsum('i,ij->ij', mueff * fd, mesh.d2dy2) \
        + np.einsum('i,ij->ij', (mesh.ddy @ mueff) * fd, mesh.ddy)

    # implicitly treated source term
    np.fill_diagonal(A, A.diagonal() - r * e / k / fs)

    # Right-hand-side
    b = -Pk[1:-1]

    # Wall boundary conditions
    k[0] = k[-1] = 0.0

    # Solve TKE
    k = sol_eqn(k * fs, A, b, 0.7) / fs
    k[1:-1] = np.maximum(k[1:-1], 1.e-12)

    return mut, k, e, Pk


def sst(mesh, u, k, om, r, mu ):
    n = mesh.nPoints
    # model constants
    sigma_k1 = 0.85
    sigma_k2 = 1.0
    sigma_om1 = 0.5
    sigma_om2 = 0.856
    beta_1 = 0.075
    beta_2 = 0.0828
    betaStar = 0.09
    a1 = 0.31
#    alfa_1 = beta_1 / betaStar - sigma_om1 * 0.41 ** 2.0 / betaStar ** 0.5
#    alfa_2 = beta_2 / betaStar - sigma_om2 * 0.41 ** 2.0 / betaStar ** 0.5
    alfa_1 = 5.0/9.0 #beta_1 / betaStar - sigma_om1 * 0.41 ** 2.0 / betaStar ** 0.5
    alfa_2 = 0.44 #beta_2 / betaStar - sigma_om2 * 0.41 ** 2.0 / betaStar ** 0.5
    # Relaxation factors
    underrelaxK = 0.6
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
    sigma_k = sigma_k1 * bF1 + (1 - bF1) * sigma_k2
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

    # ---------------------------------------------------------------------
    # k-equation

    # effective viscosity
    mueff = mu + sigma_k * mut
    fs = fd = np.ones(n)

    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = np.einsum('i,ij->ij', mueff * fd, mesh.d2dy2) \
        + np.einsum('i,ij->ij', (mesh.ddy @ mueff) * fd, mesh.ddy)

    # implicitly treated source term
    np.fill_diagonal(A, A.diagonal() - betaStar * r * om / fs)

    # Right-hand-side
    Pk = np.minimum(mut * strMag * strMag, 20 * betaStar * k * r * om)
    b = -Pk[1:-1]

    # Wall boundary conditions
    k[0] = k[-1] = 0.0

    # Solve
    k = sol_eqn(k * fs, A, b, underrelaxK) / fs
    k[1:-1] = np.maximum(k[1:-1], 1.e-12)
    
    e = betaStar*k*om
    return mut, k, e, Pk, om


def v2f(mesh, u, k, e, v2, r, mu):
    n = mesh.nPoints
    f = np.zeros(n)
    # Model constants
    cmu = 0.22
    sigk = 1.0
    sige = 1.3
    Ce2 = 1.9
    Ct = 6
    Cl = 0.23
    Ceta = 70
    C1 = 1.4
#    C1 = 2.2
    C2 = 0.3

    # Relaxation factors
    underrelaxK = 0.6
    underrelaxE = 0.6
    underrelaxV2 = 0.6

    # Time and length scales, eddy viscosity and turbulent production
    Tt = np.maximum(k / e, Ct * np.power(mu / (r * e), 0.5))
    Lt = Cl * np.maximum(np.power(k, 1.5) / e, Ceta * np.power(np.power(mu / r, 3) / e, 0.25))
    mut = np.maximum(cmu * r * v2 * Tt, 0.0)
    Pk = mut * np.power(mesh.ddy @ u, 2.0)

    # ---------------------------------------------------------------------
    # f-equation

    # implicitly treated source term
    A = np.einsum('i,ij->ij', Lt * Lt, mesh.d2dy2)
    np.fill_diagonal(A, A.diagonal() - 1.0)

    # Right-hand-side
    vok = v2[1:-1] / k[1:-1]
    rhsf = ((C1 - 6) * vok - 2 / 3 * (C1 - 1)) / Tt[1:-1] - C2 * Pk[1:-1] / (r[1:-1] * k[1:-1])

    # Solve
    f = sol_eqn(f, A, rhsf, 1)
    f[1:-1] = np.maximum(f[1:-1], 1.e-12)

    # ---------------------------------------------------------------------
    # v2-equation:

    # effective viscosity and pre-factors for compressibility implementation
    mueff = mu + mut
    fs = fd = np.ones(n)

    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = np.einsum('i,ij->ij', mueff * fd, mesh.d2dy2) \
        + np.einsum('i,ij->ij', (mesh.ddy @ mueff) * fd, mesh.ddy)

    # implicitly treated source term
    np.fill_diagonal(A, A.diagonal() - 6.0 * r * e / k / fs)

    # Right-hand-side
    b = -r[1:-1] * k[1:-1] * f[1:-1]

    # Wall boundary conditions
    v2[0] = v2[-1] = 0.0

    # Solve
    v2 = sol_eqn(v2 * fs, A, b, underrelaxV2) / fs
    v2[1:-1] = np.maximum(v2[1:-1], 1.e-12)

    # ---------------------------------------------------------------------
    # e-equation

    # effective viscosity
    mueff = mu + mut / sige
    fs = np.ones(n)
    fd = np.ones(n)

    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = np.einsum('i,ij->ij', mueff * fd, mesh.d2dy2) \
        + np.einsum('i,ij->ij', (mesh.ddy @ mueff) * fd, mesh.ddy)

    # implicitly treated source term
    np.fill_diagonal(A, A.diagonal() - Ce2 / Tt * r / fs)

    # Right-hand-side
    Ce1 = 1.4 * (1 + 0.045 * np.sqrt(k[1:-1] / v2[1:-1]))
    b = -1 / Tt[1:-1] * Ce1 * Pk[1:-1]

    # Wall boundary conditions
    e[0] = mu[0] * k[1] / r[0] / np.power(mesh.y[1] - mesh.y[0], 2)
    e[-1] = mu[-1] * k[-2] / r[-1] / np.power(mesh.y[-1] - mesh.y[-2], 2)

    # Solve
    e = sol_eqn(e * fs, A, b, underrelaxE) / fs
    e[1:-1] = np.maximum(e[1:-1], 1.e-12)

    # ---------------------------------------------------------------------
    # k-equation

    # effective viscosity
    mueff = mu + mut / sigk
    fs = fd = np.ones(n)


    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = np.einsum('i,ij->ij', mueff * fd, mesh.d2dy2) \
        + np.einsum('i,ij->ij', (mesh.ddy @ mueff) * fd, mesh.ddy)

    # implicitly treated source term
    np.fill_diagonal(A, A.diagonal() - r * e / k / fs)

    # Right-hand-side
    b = -Pk[1:-1]

    # Wall boundary conditions
    k[0] = k[-1] = 0.0

    # Solve
    k = sol_eqn(k * fs, A, b, underrelaxK) / fs
    k[1:-1] = np.maximum(k[1:-1], 1.e-12)

    return mut, k, e, Pk, v2