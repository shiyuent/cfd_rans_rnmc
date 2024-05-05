import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps

from subfunc.mesh import Mesh
from subfunc.subfuncs import grad_w
from subfunc.metric import rmse_score, mape_score
from step02_sol_temp.sol_temp_eq import sol_temper_gep


caseRetau = np.array([395, 375, 370, 356, 405, 420, 438])
caseRa = np.array([0,2.1629e6,3.9969e6, 9.3343e6, 1.8634e6, 3.5697e6, 8.2904e6])

# input dns_data source
rans_mod = 'mk' # 'mk' or 'std' or 'dns.std' or 'dns.mk'
# heat flux models, if "base", SGDH, else "gep" models
hf_mod = "gep" # 'gep' or "base"

#case_list = [3, 4, 6]
#case_list = [0, 0, 0, 0, 0, 0, \
case_list = [0, 1, 2, 3, 4, 5, 6, \
             3, 4, 6, 3, 4, 6, \
             3, 4, 6, 3, 4, 6, \
             0, 1, 2, 3, 4, 5, 6, \
             0, 1, 2, 3, 4, 5, 6, \
             0, 1, 2, 3, 4, 5, 6, \
             0, 1, 2, 3, 4, 5, 6, \
             0, 1, 2, 3, 4, 5, 6, \
             0, 1, 2, 3, 4, 5, 6, \
             0, 1, 2, 3, 4, 5, 6, \
             0, 1, 2, 3, 4, 5, 6, \
             0, 1, 2, 3, 4, 5, 6, \
             0, 1, 2, 3, 4, 5, 6, \
             0, 1, 2, 3, 4, 5, 6, \
             0, 1, 2, 3, 4, 5, 6]
#input_mod_list =['dns', 'dns.std', 'dns.mk', 'dns', 'std', 'mk', \'std', 
input_mod_list =['mk','mk', 'mk', 'mk', 'mk','mk','mk', \
                 'std', 'std', 'mk', 'mk', 'mk', \
                 'std', 'std', 'std', 'mk', 'mk', 'mk',\
                 'mk','mk', 'mk', 'mk', 'mk','mk','mk', \
                 'mk','mk', 'mk', 'mk', 'mk','mk','mk', \
                 'std', 'std', 'std','std','std', 'std', 'std', \
                 'mk','mk', 'mk', 'mk', 'mk','mk','mk', \
                 'lb','lb', 'lb', 'lb', 'lb','lb','lb', \
                 'lb','lb', 'lb', 'lb', 'lb','lb','lb', \
                 'lb','lb', 'lb', 'lb', 'lb','lb','lb', \
                 'lb','lb', 'lb', 'lb', 'lb','lb','lb', \
                 'mk','mk', 'mk', 'mk', 'mk','mk','mk', \
                 'mk','mk', 'mk', 'mk', 'mk','mk','mk', \
                 'mk','mk', 'mk', 'mk', 'mk','mk','mk', \
                 'mk','mk', 'mk', 'mk', 'mk','mk','mk'] 
# hf_mod_list = ['base', 'base', 'base', 'gep', 'base', 'base', \'base'
hf_mod_list = ['gep', 'gep', 'gep', 'gep','gep', 'gep', 'gep', \
               'base', 'base', 'base', 'base', 'base', \
               'gep', 'gep', 'gep', 'gep', 'gep', 'gep',\
               'gep', 'gep', 'gep', 'gep','gep', 'gep', 'gep', \
               'base', 'base', 'base', 'base', 'base', 'base', 'base', \
               'base', 'base', 'base', 'base', 'base', 'base', 'base', \
               'gep', 'gep', 'gep', 'gep','gep', 'gep', 'gep', \
               'base', 'base', 'base', 'base', 'base', 'base', 'base',\
               'gep', 'gep', 'gep', 'gep','gep', 'gep', 'gep', \
               'gep', 'gep', 'gep', 'gep','gep', 'gep', 'gep', \
               'gep', 'gep', 'gep', 'gep','gep', 'gep', 'gep', \
               'gep', 'gep', 'gep', 'gep','gep', 'gep', 'gep', \
               'gep', 'gep', 'gep', 'gep','gep', 'gep', 'gep', \
               'gep', 'gep', 'gep', 'gep','gep', 'gep', 'gep', \
               'gep', 'gep', 'gep', 'gep','gep', 'gep', 'gep']

#j = 0  # bigger than zero
j = -1
j_stop = 6
## Set input and output folder + filename
ifolder = 'step02_sol_temp/input/'
ofolder = 'step02_sol_temp/output/'
## Set key parameters
# re-define key parameters
height = 2  # channel height
Pr = 0.709 


## Generate 1D mesh
n = 100  # number of mesh points
fact = 6  # streching factor and stencil for finite difference discretization
mesh = Mesh(n, height, fact, 1)

#for i in case_list:
#for i in range(3, 4,1):
#for i in range(4, 5,1):
for i in range(0,7,1):

    j = j + 1
    print('this is case:', j)
    if j > j_stop:
        break
    
    # input dns_data source
    rans_mod = input_mod_list[j-1]
    # heat flux models, if "base", SGDH, else "gep" models
    hf_mod = hf_mod_list[j-1]

    outStr = rans_mod + '.case' + str(i)
    paraStr = rans_mod + '.'+ hf_mod + '.'+ 'case' + str(i)
    figStr = rans_mod + '_'+ hf_mod + '_case' + str(i)

    ifilename = 'input.temp.' + outStr
    ofilename = 'output' + str(j) +'.temp.' + outStr
    ofilepara = 'result' + str(j) +'.temp.' + paraStr 
    ## Import data-source, initial the whole field
    ifile = ifolder + ifilename
    print("name of input file ：", ifile)
    ofile = ofolder + ofilename
    print("name of output file ：", ofile)
    ofilepara = ofolder + ofilepara
    print("name of outpara file ：", ofilepara)
    
    # Set parameter in different case
    ReTau =  caseRetau[i] # or input a ReTau instead
    mu_ray = 1.0 / ReTau  # the mu for Rayleigh number calculation
    # Gr = Rayleigh /Prantal number
    Gr= caseRa[i]/Pr 
    gbeta  = Gr* (mu_ray**2)/(height**3)   # in temp. equation, \delta_T
    print("gbeta   = ", gbeta)  # buoyancy effect

    # load dns data, interp. as RANS input
    input_dns = np.loadtxt(ifile,skiprows=4)
    u_in =   np.interp(mesh.y, input_dns[:,0], input_dns[:,1])
    c_in = np.interp(mesh.y, input_dns[:,0], input_dns[:,2])
    k_in = np.interp(mesh.y, input_dns[:,0], input_dns[:,3])
    e_in = np.interp(mesh.y , input_dns[:,0], input_dns[:,4])
    nut_in = np.interp(mesh.y , input_dns[:,0], input_dns[:,5])
    wc_in = np.interp(mesh.y , input_dns[:,0], input_dns[:,6])
        
    c2, turbpr2, k2, ep2, wc2 = sol_temper_gep(mesh, u_in, k_in, e_in, nut_in,\
                                               c_in, Pr, ReTau, hf_mod)
   
    utau=np.sqrt(mu_ray*grad_w(mesh, u_in))
    utau_avg = utau[2]
    print('utau is:', utau)
    retau_rans =height/2.*utau/mu_ray
    print('retau is:', retau_rans)
    
    c_err = rmse_score(c2, c_in)
    wc_err= rmse_score(wc2/utau_avg, wc_in)
    print(c_err, wc_err) # give rmse scores

    # wirte out rmse errors of    
    with open(ofilepara, 'a') as out_file:
        out_file.write("current utau:\n")
        out_file.write(repr(utau)+'\n')   
        out_file.write("current retau:\n")
        out_file.write(repr(retau_rans)+'\n')   
        out_file.write("rmse score of mean temperature profile:\n")
        out_file.write(repr(c_err)+'\n')
        out_file.write("rmse score of heat flux profile:\n")
        out_file.write(repr(wc_err)+'\n')
    out_file.close()


    nu_dns= height*grad_w(mesh, c_in)
    nu_rans = height*grad_w(mesh, c2)
    print('nu_01 is:', nu_dns, nu_rans)
    
    nu_err01 = mape_score(nu_rans, nu_dns)
    
    t_bulk=simps(c2,mesh.y)/height
    nu_hot = 2*nu_rans[1]/(0.5-t_bulk)*mesh.y[np.argmax(u_in)]
    nu_cold = 2*nu_rans[2]/np.abs(-0.5-t_bulk)*(height-mesh.y[np.argmax(u_in)])

    t_bulk_in=simps(c_in,mesh.y)/height
    nu_hot_dns = 2*nu_dns[1]/(0.5-t_bulk_in)*mesh.y[np.argmax(u_in)]
    nu_cold_dns = 2*nu_dns[2]/np.abs(-0.5-t_bulk_in)*(height-mesh.y[np.argmax(u_in)])
    nu_err02_hot = mape_score(nu_hot, nu_hot_dns)
    nu_err01_cold = mape_score(nu_cold, nu_cold_dns)

    print('nu_02_hot is:', nu_hot_dns, nu_hot)        
    print('nu_02_cold is:', nu_cold_dns, nu_cold)        

    # wirte out nusselt number, and mape errors from the models
    with open(ofilepara, 'a') as out_file:
        out_file.write("1st defination, nu_rans:\n")
        out_file.write(repr(nu_rans)+'\n')  
        out_file.write("1st defination, nu_dns:\n")
        out_file.write(repr(nu_dns)+'\n')  
        out_file.write("mape score of nu-01:\n")
        out_file.write(repr(nu_err01)+'\n')
        
        out_file.write("2nd defination, nu_rans:\n")
        out_file.write(repr(nu_hot)+ ',' + repr(nu_cold)+ '\n')  
        out_file.write("2nd defination, nu_dns:\n")
        out_file.write(repr(nu_hot_dns)+ ',' + repr(nu_cold_dns)+ '\n')  
        out_file.write("mape score of nu-02:\n")
        out_file.write(repr(nu_err02_hot)+ ','+ repr(nu_err01_cold) +'\n')
    out_file.close()


    retau395=np.c_[mesh.y, c2, wc2/utau_avg, turbpr2]
    np.savetxt(ofile,retau395, delimiter=',')  
    
    n = mesh.nPoints
    ## post-processing
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure()
    ax = plt.subplot(1 ,1 ,1)
    line1 = ax.plot(mesh.y[1:n], c2[1:n] , 'r-', linewidth=2)
    line2 = ax.plot(input_dns[:,0], input_dns[:,2] , 'k--', linewidth=2)
    plt.xlabel(r"$y/\delta$", fontsize=16)
    plt.ylabel(r"$\frac{T-T_1}{\Delta T}$", fontsize=16)
    plt.legend(('rans','dns'))
    plt.savefig(ofolder+'Fig' + str(j) + '_c_'+ figStr + '.pdf',bbox_inches='tight',transparent=True)
    plt.savefig(ofolder+'Fig' + str(j) + '_c_'+ figStr + '.png',dpi=300,bbox_inches='tight')
    
    plt.figure()
    ax = plt.subplot(1 ,1 ,1)
    line1 = ax.plot(mesh.y[1:n], wc2[1:n]/utau_avg, 'r-', linewidth=2)
    line2 = ax.plot(input_dns[:,0], input_dns[:,6] , 'k--', linewidth=2)
    plt.xlabel(r"$y/\delta$", fontsize=16)
    plt.ylabel(r"$\frac{wc}{u_\tau \Delta T}$", fontsize=16)
    plt.legend(('rans','dns'))
    plt.savefig(ofolder+'Fig' + str(j) + '_wc_'+ figStr + '.pdf',bbox_inches='tight',transparent=True)
    plt.savefig(ofolder+'Fig' + str(j) + '_wc_'+ figStr + '.png',dpi=300,bbox_inches='tight')