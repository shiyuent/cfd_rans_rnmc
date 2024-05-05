import numpy as np
from subfunc.subfuncs import sol_eqn

def sol_temper_gep(mesh, u, k, ep, mut, temperature, Pr, ReTau, hf_model):
    """
    solve Temperature equation, isothermal bc, with force, natural, and mixed conveciton cases
    """
    n = mesh.nPoints
    # molecular thermal conductivity:
    lam = np.ones(n) / (ReTau * Pr)

    # turbulent prandtl: assume = 0.9
    # cal I1-J1 based on u profile and mesh 
    
    grad_u = mesh.ddy @ u
    eps = np.maximum(ep, 1e-8)

#    temperature = np.zeros(n)  # temperature
#    mut, om = sst_mut(mesh, u, k, ep, r, mu)
    
    res_c = 1.0e20
    
    iterations = 0
    print("Start iterating")
    
    while res_c > 1.e-6 and iterations < 1e8:
        # Solve temperature:  d/dy[(lam+mut/PrT)dTdy] = -VolQ/ReTau/Pr
        # diffusion matrix: lamEff*d2phi/dy2 + dlamEff/dy dphi/dy
        # Isothermal BC
        temperature[0] = 0.5
        temperature[-1] = -0.5
        b =  np.zeros(n - 2)  # no heat source in for whole flow    
        
        if hf_model == "base":
            turbpr = 0.90*np.ones(n)
            a = np.einsum('i,ij->ij', mesh.ddy @ (lam + mut/turbpr), mesh.ddy) + np.einsum('i,ij->ij', lam + mut/turbpr,mesh.d2dy2)                                                                                                                                                    
        elif hf_model == "gep":
            grad_c = mesh.ddy @ temperature
            ttau = np.abs(grad_c[0]/ReTau/Pr) # utau =1 
            
            I1  = np.power(0.09*k/eps *1/ReTau * grad_u, 2.0) *0.5
            J1  = np.power(0.09*np.power(k,1.5) /eps * 1/ReTau * grad_c /ttau , 2.0)
            
            # dns-case-00
     #       fgep = 1.17677 *np.ones(n)+ i1*(-1.27623 + (0.142881 - 0.097*i1)*i1 + 0.194*j1) + 0.018818*j1
           # fgep = 1.15*np.ones(n)- 0.795*i1 + 0.097*(0.097*np.ones(n) + i1)*(i1 + j1)

            # mk-case-03 
          #  fgep = 0.947*np.ones(n) - i1 - 2.0*(-0.43*np.ones(n) + i1)*(-0.089 + i1)*j1
            # mk-case-04
          #  fgep = np.ones(n) + i1*(3*np.ones(n)  - 2*j1)
            # mk-case-06
          #  fgep = 1.05686*np.ones(n)  + 0.56481*i1 - 0.18827*j1
            
            # std-case-03 
            #fgep = 0.85* np.ones(n)  - 3*i1
            # std-case-04
           # fgep = np.ones(n) + i1*(-0.928*np.ones(n) - 4.56*i1 - 8.0*j1)
            # std-case-06
      #      fgep = 1.089*np.ones(n) + i1 * (-4. - 4.0*i1 + i1*(-2. + i1 + j1))           

      # lb-case-03
          #  fgep=0.93815*np.ones(n) + (-0.911 + 0.822*i1)*i1      
      # lb-case-04
           # fgep = 1.144*np.ones(n) - 0.097*j1
      # lb-case-06
      #      fgep = np.ones(n) + 0.0645*(-1.294 + i1)*j1
      
      # loop-models, c prof. mini.
            # fgep = (i1**2)*(i1 - 3*j1 - 1.0134478668118685) + 1.0
            #fgep = eval('-0.43037873274483895*i1*(-2*i1 + 1.5830691340670295*j1 + 0.43037873274483895) + 1.0')
            
            # loop-c-test-1, run 75-81
          #  fgep = eval('((1.0)+(((((I1)*(((((I1)*(J1)))+(((I1)-(((((0.43037873274483895)+(J1)))+(((((I1)*(J1)))-(I1)))))))))))*(0.43037873274483895))))')
          #  fgep = eval('((1.0)+(((((I1)*(((((-0.15269040132219058)*(J1)))+(((I1)-(((((0.43037873274483895)+(J1)))+(((I1)*(J1)))))))))))*(0.43037873274483895))))')
            # loop-nu-test-1, run 89-95
          #  fgep = eval('((1.0)+(((((I1)*(J1)))-(0.0976270078546495))))')
          # loop-nu-test2, run 96...
          #  fgep = eval('((1.0)+(((((((J1)-(((0.43037873274483895)-(J1)))))*(J1)))-(J1))))')
            # fgep = eval('1.0 + 2.0*(-0.715189 + J1)*J1')
       # test natural convection models
       #     fgep = eval('1.1 - 0.02715*I1 - 4.4757*J1') ---bad
       # weight-1 
       #     fgep =eval('1.0453 + 0.4791*I1 - 0.16455*J1') --bad
       #     fgep = eval('1.0441 + 0.45*I1 - 0.15*J1')
        #    fgep = eval('1.0615 + 0.45*I1 - 0.18075*J1')
    # weight-4, important models, case-6, Ri050
    #        fgep = eval('1.0441 + 0.45*I1 - 0.15*J1')
    # weight-4, case-1, Ri018
    #        fgep = eval('1.14151 + I1*(0.387-0.387*J1)-0.141513*J1')
    # weight-4, case-3, Ri094
            # fgep = eval('0.966 - I1 - 2.0*(-0.43 + I1)*(I1 - 0.089*J1)*J1')    
    # c10nu, only models that is good, so called multi-objective modelling 
#          fgep = eval('0.970187 - 0.305381*(I1**2)')     
    
#           fgep = eval('1. + (0.089 + I1)*(0.15 + I1)*(-1.856 + I1 * (-2. + J1))')
   # using mixed data
   #         fgep =eval('1.0+(I1-0.43*J1)*(0.15+(-0.624+I1)*I1+0.097*J1)')  
   # dcdy_case6-1
   #         fgep=eval('1.08977 + I1*(-1.32716 - 0.84731*I1 + 0.0897664*J1)')
   #dcdy_case6-2
            fgep=eval('1.09853 + I1*(-1.17953 - 0.0897664*J1)')
    #fgep = np.ones(n)
            turbpr = 1/fgep
            a = np.einsum('i,ij->ij', mesh.ddy @ (lam + mut*fgep), mesh.ddy) + np.einsum('i,ij->ij', lam + mut*fgep,mesh.d2dy2)                                                                                                                                                    
        
        # Solve
        c_old = temperature.copy()
        temperature = sol_eqn(temperature, a, b, 0.95)

        res_c = np.linalg.norm(temperature - c_old) / n
        # Printing residuals
        if iterations % 500 == 0: print("iteration: ", iterations, ", Residual(c) = ", res_c)
        iterations = iterations + 1
    
    wc = -1* (mut /turbpr) * (mesh.ddy @ temperature)
    return temperature, turbpr, k ,ep, wc