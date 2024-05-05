import numpy as np

def sol_eqn(x, a, b, omega):
    n = np.size(x)
    x_new = x.copy()
    # add boundary conditions
    b = b - x[0] * a[1:n - 1, 0] - x[n-1] * a[1:n - 1, n - 1]
    # perform under-relaxation
    b[:] = b[:] + (1 - omega) / omega * a.diagonal()[1:-1] * x[1:-1]
    np.fill_diagonal(a, a.diagonal() / omega)
    # solve linear system
    x_new[1:-1] = np.linalg.solve(a[1:-1, 1:-1], b)
    return x_new

def grad_w(mesh, phi):
    '''
    grad_w to get the wall value, (gradient terms)
    gradw[0], gradient at hot wall
    gradw[1], gradient at cold wall
    gradw[2], mean of both hot and cold wall
    '''
    grad_phi = mesh.ddy @ phi
    gradw = np.zeros(3)
    gradw[0]=np.abs(grad_phi[0])
    gradw[1]=np.abs(grad_phi[-1])
    gradw[2] = gradw[0]/2. + gradw[1]/2.
    return gradw