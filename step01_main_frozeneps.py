import matplotlib.pyplot as plt
import numpy as np

from subfunc.mesh import Mesh
#from step01_frozen_eps.sol_frozeneps import kep_frozen_eps 
#from step01_frozen_eps.sol_frozeneps import kepMK_frozen_eps
from step01_frozen_eps.sol_frozeneps import kepLB_frozen_eps
# from step01_frozen_eps.sol_frozeneps import sst_frozen_eps # bug!

caseRetau = np.array([395, 375, 370, 356, 405, 420, 438])
## Set input and output folder + filename
ifolder = 'step01_frozen_eps/init_get_input/'
ofolder = 'step01_frozen_eps/output/'
## Set key parameters
# re-define key parameters
height = 2  # channel height
Pr = 0.709 
## Generate 1D mesh
n = 200  # number of mesh points
fact = 6  # streching factor and stencil for finite difference discretization
mesh = Mesh(n, height, fact, 1)

for i in range(caseRetau.shape[0]):
    ifilename = 'input.data.case' + str(i)
    ofilename = 'output.data.lb.case' + str(i)
    ReTau =  caseRetau[i] # or input a ReTau instead
    ## Import data-source, initial the whole field
    ifile = ifolder + ifilename
    print("name of input file ", ifile)
    ofile = ofolder + ofilename
    print("name of input file ", ofile)
    ## Load DNS data
    input_dns = np.loadtxt(ifile,skiprows=0)
    u_in = np.interp(mesh.y, input_dns[:,0], input_dns[:,1])
    k_in = np.interp(mesh.y , input_dns[:,0], input_dns[:,2])
    eps = np.interp(mesh.y , input_dns[:,0], input_dns[:,3])
    eps = eps*ReTau
    
    iterations = 0
    print("Start iterating")
    res_ep = 1.0e20    
    while res_ep > 1.e-6 and iterations < 1e7: 
        eps_old = eps.copy()
     #   mut, eps = kep_frozen_eps(mesh, u_in, k_in, eps, ReTau)
#        mut, eps = kepMK_frozen_eps(mesh, u_in, k_in, eps, ReTau)  
        mut, eps = kepLB_frozen_eps(mesh, u_in, k_in, eps, ReTau)   
     #   mut, eps = sst_frozen_eps(mesh, u_in, k_in, eps, ReTau)   

        res_ep = np.linalg.norm(eps - eps_old) / n
        # Printing residuals
        if iterations % 100 == 0: print("iteration: ", iterations, ", Residual(eps) = ", res_ep)
        iterations = iterations + 1
    
    ## output data to a ascii file
    mu = 1/ReTau   
    utau_hot=np.sqrt(mu*abs((u_in[1]-u_in[0])/(mesh.y[1]-mesh.y[0])))
    utau_cold=np.sqrt(mu*abs((u_in[-2]-u_in[-1])/(mesh.y[-2]-mesh.y[-1])))
    utau_avg=(utau_hot+utau_cold)/2.
    print(utau_hot, utau_cold,utau_avg)
    
    
    #out_data=np.c_[mesh.y, eps, mut, om]
    out_data=np.c_[mesh.y, eps/ReTau/np.power(utau_avg, 4.), mut/utau_avg]
    np.savetxt(ofile, out_data, delimiter=',')  
    
    ## post-processing
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(12.000000,9.000000))
    ax = plt.subplot(2,2, 1)  
    ax.plot(input_dns[:,0], input_dns[:,1] , '--b', linewidth=2)
    plt.xlabel(r"$y/\delta$", fontsize=16)
    plt.ylabel(r"$U/u_{\tau}$", fontsize=16)
    plt.legend(('dns'))
    
    ax = plt.subplot(2,2, 2)  
    ax.plot(input_dns[:,0], input_dns[:,2] , '--b', linewidth=2)
    plt.xlabel(r"$y/\delta$", fontsize=16)
    plt.ylabel(r"$k/u^2_{\tau}$", fontsize=16)
    plt.legend(('dns'))
    
    
    ax = plt.subplot(2,2, 3)  
    ax.plot(mesh.y, mut/utau_avg , '-r', linewidth=2)
    plt.xlabel(r"$y/\delta$", fontsize=16)
    plt.ylabel(r"$\nu_t/(u_\tau \delta)$", fontsize=16)
    plt.legend(('rans'))
    
    ax = plt.subplot(2,2, 4)  
    l1=ax.plot(mesh.y, eps/ReTau/np.power(utau_avg, 4.) , '-r', linewidth=2)
    l2=ax.plot(input_dns[:,0], input_dns[:,3] , '--b', linewidth=2)
    
    plt.xlabel(r"$y/\delta$", fontsize=16)
    plt.ylabel(r"$\epsilon \nu / u^4_\tau$", fontsize=16)
    plt.legend(('rans', 'dns'))
    #plt.savefig(ofolder+'/figures/'+'Fig01_case'+str(i)+'_mk_all.pdf',bbox_inches='tight',transparent=True)
    #plt.savefig(ofolder+'/figures/'+'Fig01_case'+str(i)+'_mk_all.png',dpi=300,bbox_inches='tight')
#    plt.savefig(ofolder+'/figures/'+'Fig03_case'+str(i)+'_std_all.pdf',bbox_inches='tight',transparent=True)
#    plt.savefig(ofolder+'/figures/'+'Fig03_case'+str(i)+'_std_all.png',dpi=300,bbox_inches='tight')
    plt.savefig(ofolder+'/figures/'+'Fig05_case'+str(i)+'_lb_all.pdf',bbox_inches='tight',transparent=True)
    plt.savefig(ofolder+'/figures/'+'Fig05_case'+str(i)+'_lb_all.png',dpi=300,bbox_inches='tight')
        
    plt.figure()
    plt.semilogx(mesh.y*ReTau, eps/ReTau/np.power(utau_avg, 4.),'-r', linewidth=2)
    plt.semilogx(input_dns[:,0]*ReTau, input_dns[:,3] , '--b', linewidth=2)
    plt.xlim(.1,ReTau)
    plt.xlabel(r"$y^+$", fontsize=16)
    plt.ylabel(r"$\epsilon \nu / u^4_\tau$", fontsize=16)
    #plt.savefig(ofolder+'/figures/'+'Fig02_case'+str(i)+'_mk_eps.pdf',bbox_inches='tight',transparent=True)
    #plt.savefig(ofolder+'/figures/'+'Fig02_case'+str(i)+'_mk_eps.png',dpi=300,bbox_inches='tight')
#    plt.savefig(ofolder+'/figures/'+'Fig04_case'+str(i)+'_std_eps.pdf',bbox_inches='tight',transparent=True)
#    plt.savefig(ofolder+'/figures/'+'Fig04_case'+str(i)+'_std_eps.png',dpi=300,bbox_inches='tight')
    plt.savefig(ofolder+'/figures/'+'Fig06_case'+str(i)+'_lb_eps.pdf',bbox_inches='tight',transparent=True)
    plt.savefig(ofolder+'/figures/'+'Fig06_case'+str(i)+'_lb_eps.png',dpi=300,bbox_inches='tight')
    