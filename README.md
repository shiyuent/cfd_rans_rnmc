## RANS-solver for mixed (natural, forced) convection 

**rnmc** 
stands for RANS-solver in a vertical channel with isothermal boundary condition, it can be used in cases such as force, natural, and mixed conveciton. Sub folders contains solver for GEP based training procedures. Here, only constant fluidproperties: $\rho = 1$; $\mu = 1/Re_\tau$

The present data and the associated source code are freely available under the GNU GPL v3 licence, EXCEPT for the *tex* file in the folder [article](/article/). They correspond to the paper entitled *name of paper* we have published in the [International Journal of Heat and Mass Transfer](https://www.journals.elsevier.com/international-journal-of-heat-and-mass-transfer)). The preprint of the paper is freely available on place.

### to do list
- [x] step00, verificaiton RANS code for mixed convection cases. (solve all the equation)
- [x] step01, solve $\epsilon$ equation, generate RANS-corrected turbulent dissipation rate and eddy viscosity $\nu_t$
- [x] step02, solve temperature equation, test baseline (SGDH, standard gradient diffusion hypothesis) heat flux models (compare step01's $\nu_t$ with DNS in advance)
- [x] step03, using *GEP* (genetic expression programming, **eve-dev**, Jack's code) to train *GEP* models
- [x] step04, solve temperature equation, test *GEP* heat flux models.

### prerequisites
- python 3.5
- matlab 

### folders and files
1. folder [main](/) 
   - step00_main_rans.py, solve all rans equation, data and figures can be find
   - step01_main_frozeneps.py, run this file, solve epsilon equation, data and figures can be find [output](/step01_frozen_eps/output/) 
   - step02_main_temp.py, run this file, solve temperature equaiton, data and figures can be find [output](/step02_sol_temp/output/) 

2. folder [step00_all_rans](/step00_all_rans/) 
   - sol_eddy_visco.py, turbulence models, compute eddy visocisty.
   - sol_temperature.py, RANS sovler for temperture equation.
   - sol_velo.py, RANS for velocity field, with various convection in a vertical channel 
  
    **Build-in models**
    
     - no, without turbulence model, laminar
     - kep, standard k-epsilon model
     - sst, Menter's SST k-omega model (Menter, 1995)
     - kepXu, (Xu, 1995)
     - kepMK, k-epsilon model based on Myong and Kasagi (1993)
  

    **Build-in flow type**

      - "fc", force convection in a channel
      - "vnc", vertical natural convection
      - "mxc", vertical mixed convection

3. folder [step01_frozen_eps](/step01_frozen_eps/) 
   - folder [init_get_input](/step01_frozen_eps/init_get_input/): generate input for DNS dataset ($Re_\tau = 395$)
     * case 0, forced convection, DNS data is extraced from  [MKM-1999](https://turbulence.oden.utexas.edu/data/MKM/) channel data, file ('chan_395_dns.data') contains $y$, $U/u_\tau$, $k/u^2_\tau$, $\epsilon \nu /u^4_\tau$
      * case i, mixed convection, DNS data is provided by [Dr. Duncan Sutherland](https://www.researchgate.net/profile/D_Sutherland), stored in **Matlab** *re395_ensemble.mat*. Seven cases here, with global friction Reynolds number *caseRetau = np.array([395, 375, 370, 356, 405, 420, 438]*. for all *input.data.casei*, only three first column are useful, say $y$, $U/u_\tau$, $k/u^2_\tau$, the fouth one is $\epsilon \nu /u^4_\tau$ from forced convection case (case 0).
   - sol_frozeneps.py, solve epsilon equation, two different frozen eps subfunctions are provided
      * kep_frozen_eps, eps equation in standard $k-\epsilon$ model
      * kepMK_frozen_eps, eps equation in Myong and Kasagi (1993)'s low Reynolds number $k-\epsilon$ model

4. folder [step02_sol_temp](/02_sol_temp/) 
   - folder [input](/step02_sol_temp/input/): generate input for DNS dataset ($Re_\tau = 395$)
   - sol_temp_eq.py, solve the temperature equation

5. folder [subfunc](/subfunc/) 
   - mesh.py, a 1D strenched mesh gird.
   - subfuncs.py 
     - func: sol_eqn, auxiliary function to solve linear system with under-relaxation, The linear system is $Ax = b$
      Using an under-relaxation parameter $\omega$, the system can also be written as,
      $$
      \left(A - \frac{1-\omega}{\omega}A_{ii}\right)x_{new} = b - \frac{1-\omega}{\omega}A_{ii}x_{old} 
      $$
     - func: grad_w, to get the wall value, (gradient terms)
   - metric.py
     - func: rmse, root-mean-squre error
     - func: mape, mean-absolute-percenatge-error

### reference 
- code, [RANS_Channel](https://github.com/Fluid-Dynamics-Of-Energy-Systems-Team/RANS_Channel/)
- papers, in *ref.bib*
    **turbulence models**
    - kepMK: Myong, H.K. and Kasagi, N., "A new approach to the improvement of k-epsilon turbulence models for wall bounded shear flow", JSME Internationla Journal, 1990.
    - sst: Menter, F.R., "Zonal Two equation k-omega turbulence models for aerodynamic flows", AIAA 93-2906, 1993.
    - v2f: Medic, G. and Durbin, P.A., "Towards improved prediction of heat transfer on turbine blades", ASME, J. Turbomach. 2012.
    - kepXu, Xu, W., Chen, Q., & Nieuwstadt, F. T. M. (1998). A new turbulence model for near-wall natural convection. International Journal of Heat and Mass Transfer, 41(21), 3161-3176.