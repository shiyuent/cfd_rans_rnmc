import numpy as np
from step00_all_rans.sol_eddy_visco import kep, sst, v2f
from step00_all_rans.sol_temperature import sol_temper

def sol_rans(mesh, r_in, mu_in, Pr, ReTau, gbeta, turb_model, conv_type):
    """
    solve
    """
    n = mesh.nPoints
    u = np.zeros(n)  # velocity
    temperature = np.zeros(n)  # temperature
    mut = np.zeros(n)  # eddy viscosity

    r = r_in.copy()
    mu = mu_in.copy()

    k = 0.01 * np.ones(n)  # turbulent kinetic energy
    e = 0.001 * np.ones(n)  # turbulent dissipation
    v2 = 1 / 3 * k  # wall normal turbulent fluctuations for V2F model
    om = np.ones(n)  # specific turbulent dissipation for omega in SST

    res_u = 1.0e20

    
    iterations = 0
    print("Start iterating")
    
    while res_u > 1.e-8 and iterations < 1e4: #and res_k > 1.0e-6 and res_T > 1.0e-6
        # Solve temperature:  d/dy[(lam+mut/PrT)dTdy] = -VolQ/ReTau/Pr
        r, mu, temperature = sol_temper(mesh, temperature, r,mut,Pr, ReTau)

        # Solve turbulence model to calculate eddy viscosity
        if turb_model == "kep":
            mut, k, e, Pk = kep(mesh, u, k, e, r, mu)
        elif turb_model == "sst":
            mut, k, e, Pk, om= sst(mesh, u, k, om, r, mu)
        elif turb_model == "v2f":
            mut, k, e, Pk, v2 = v2f(mesh, u, k, e, v2, r, mu)
        elif turb_model == "no":
            mut = np.zeros(n)
        else: 
            pass
        # Solve momentum equation:  0 = d/dy[(mu+mut)dudy] - 1
        # diffusion matrix: mueff*d2phi/dy2 + dmueff/dy dphi/dy
        a = np.einsum('i,ij->ij', mesh.ddy @ (mu + mut), mesh.ddy) + np.einsum('i,ij->ij', mu + mut, mesh.d2dy2)

        u_old = u.copy()
        
        if conv_type == "fc":  # force convection, velocity like channel
            u[1:n - 1] = np.linalg.solve(a[1:n - 1, 1:n - 1], -np.ones(n - 2))
        elif conv_type == 'vnc':  # vertical natural convection
            u[1:n - 1] = np.linalg.solve(a[1:n - 1, 1:n - 1], -gbeta * temperature[1:n - 1])
        elif conv_type == 'mxc':  # vertical mixed convection
            u[1:n - 1] = np.linalg.solve(a[1:n - 1, 1:n - 1], -gbeta * temperature[1:n - 1] - np.ones(n - 2))
        else:
            pass
        
        res_u = np.linalg.norm(u - u_old) / n
        
        # Printing residuals
        if iterations % 100 == 0: print("iteration: ", iterations, ", Residual(u) = ", res_u)
        iterations = iterations + 1
   # print("iteration: ", iterations, ", Residual(u) = ", residual)
    n = mesh.nPoints
    
    if turb_model == "no":
        return u, temperature
    else:
        return u,temperature, mut,k,e,Pk

