# import build-in package
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps

# import my package
from subfunc.mesh import Mesh
from step00_all_rans.sol_velo import sol_rans


## Generate mesh
height = 2  # channel height, use 2 please
n = 100  # number of mesh points
fact = 6  # streching factor and stencil for finite difference discretization
mesh = Mesh(n, height, fact, 1)

## Set key parameters
# re-define key parameters
Pr = 0.709 
ReTau = 375 #395 #177.5#156.5  # 162.8 # 177.5#162.8 #156.5  #150, 180, 395,590
r = np.ones(n)
mu = np.ones(n)/ReTau
Ray = 4.0e6  # target Rayleigh number
Gr    = Ray/Pr #9.60e5 #6.40e5        

#Gr    =0 # 1.6e6 # 6.40e5 # 9.60e5 # 1.6e6
gbeta  = Gr* (mu[0]**2)/(height**3)   # in temp. equation, \delta_T

#gbeta=0
print("gbeta   = ", gbeta)  # buoyancy effect



# "fc", force convection in a channel
# "vnc", vertical natural convection
# "mxc", vertical mixed convection

turbModel = "kep"  # turbulence model
c_type = 'mxc' # convection type, 
if turbModel == "no":
    u2, c2 = sol_rans(mesh, r, mu, Pr, ReTau, gbeta, turbModel, c_type)
else:
    u2, c2,  mut2, k2, e2, Pk2 = sol_rans(mesh, r, mu, Pr, ReTau, gbeta, turbModel, c_type)



uv2=mut2*mesh.ddy @ u2
uc2=mut2/1.*mesh.ddy @ c2

t_bulk=simps(c2,mesh.y)/height
u_bulk=simps(u2,mesh.y)/height
re_bulk=u_bulk*height/mu[0]
print(t_bulk,u_bulk,re_bulk)

utau_hot=np.sqrt(mu[0]*abs((u2[1]-u2[0])/(mesh.y[1]-mesh.y[0])))
utau_cold=np.sqrt(mu[0]*abs((u2[-2]-u2[-1])/(mesh.y[-2]-mesh.y[-1])))
utau_avg=(utau_hot+utau_cold)/2.
print(utau_hot, utau_cold,utau_avg)

nu_hot=height*abs((c2[5]-c2[0])/(mesh.y[5]-mesh.y[0]))
nu_cold=height*abs((c2[-5]-c2[-1])/(mesh.y[-5]-mesh.y[-1]))
nu_avg=(nu_hot+nu_cold)/2.
print(nu_hot, nu_cold,nu_avg)

nu_hot_1 = 2*nu_hot/(0.5-t_bulk)*mesh.y[np.argmax(u2)]
nu_cold_1 = 2*nu_cold/np.abs(-0.5-t_bulk)*(height-mesh.y[np.argmax(u2)])


retau_hot=height/2.*utau_hot/mu[0]
retau_cold=height/2.*utau_cold/mu[0]
retau_avg=(retau_hot+retau_cold)/2.
print(retau_hot, retau_cold,retau_avg)

cf_avg=2*np.power(utau_avg/u_bulk,2.)
cf_hot=2*np.power(utau_hot/u_bulk,2.)
cf_cold=2*np.power(utau_cold/u_bulk,2.)
print(cf_hot, cf_cold,cf_avg)

print(u_bulk,re_bulk,retau_avg,retau_hot, retau_cold,utau_avg, utau_hot, utau_cold, nu_hot, nu_cold,nu_avg, cf_hot, cf_cold,cf_avg)

sst=np.c_[mesh.y,mesh.y*ReTau, u2/utau_avg,c2, k2/utau_avg/utau_avg,e2/ReTau/np.power(utau_avg,4), mut2, uv2, uc2, Pk2]
#np.savetxt('sst.dat',sst, delimiter=',')  

plt.figure()
plt.plot(mesh.y, u2/utau_avg)
plt.show()