######################################
__title__          = "TTTW_timestepping.py"
__author__         = "Daniel Dauhajre"
__date__           = "September 2017"
__email__          = "ddauhajre@atmos.ucla.edu"
__python_version__ = "2.7.9"

'''

TIME-STEPPING LIBRARY
THAT TIME-STEPS EVOLUTION
EQUATIONS AND RETURNS OUTPUTTED
ARRAYS TO MAIN FOR WRITING INTO
NETCDF FILE

'''
######################################

##################################
#        IMPORT LIBRARIES
###################################
import os
import sys
import numpy as np
import scipy as sp
from pylab import *
from scipy import integrate
from copy import copy
import time as tm
from scipy.sparse import *
from scipy.sparse.linalg import spsolve
import TTTW_funcs as TTTW_func
import TTTW_kpp as KPP_mod
import multiprocessing  as MP
import time
#################################




def tstep_main(grids,IC_dict,constants_dict,tstep_dict, K_nt_dict, surf_flux_dict, choices_dict,Dh_arr):
    '''
    CALLED FROM MAIN
    
    - UNPACKS ALL INPUT DATA (initial conditions, grids, etc)
    - CALCULATES COMPUTATION INITIAL CONDITIONS WITH SEMI-IMPLICT FORWARD EULER and AB2
    - TIME STEPS UNTIL END-TIME WITH AB3 (time-staggered) SEMI-IMPLICT SCHEME

    TIME-STAGGERING (IF AB3 scheme used)
    u,v,w time-stepped at whole integer n=1,2,3,... points
    b     time-steppd at 1/2 index n=1/2, 3/2, 5/2...points


    b will be stored at whole integer n=1,2,3..points for writing netcdf by
    a simple interpolation b[n] = 0.5 * (b[n+1/2] + b[n-1/2])
 
    '''
    start_time = time.time()
    #################################
    # GET BUOYANCY ADVECTION CHOICE
    # AND BOTTOM STRESS SWITCH
    ################################
    advect_bool     = choices_dict['advect_bool']
    bot_stress_bool = choices_dict['bottom_stress']
    horiz_diff      = choices_dict['horiz_diff']

    #####################
    # GET CONSTANTS
    #####################

    f = constants_dict['f']
    rho0 = constants_dict['rho0']
    g    = constants_dict['g']
    Zob   = constants_dict['Zob']
    Dh    = constants_dict['Dh']

    #########################
    # GET TIME STEP PARAMTERS
    ########################
    dt        = tstep_dict['dt']
    tend_days = tstep_dict['tend_days']
    nt        = tstep_dict['nt']

    #############################################
    # DECLARE SOLUTIONS ARRAYS IN DICTIONARY
    ############################################
    [Ly,N] = grids['y_u_r'].shape
    var_out_dict = {}
    keys_out   = ['u', 'v', 'w', 'b', 'b_stagger','phi_total', 'Kv', 'Kt', 'hbls', 'hbbl', 'ghat']    
    shapes_out = [ [nt,Ly,N], [nt,Ly+1,N], [nt,Ly,N+1], [nt,Ly,N], [nt,Ly,N], [nt,Ly,N], [nt,Ly,N+1], [nt,Ly,N+1], [nt,Ly], [nt,Ly], [nt,Ly,N+1]] 
    for ke in range(len(keys_out)):
        var_out_dict[keys_out[ke]] = np.zeros(shapes_out[ke])


    #############################################################

    ####################################
    # SET VERTICAL MIXING TIME-SERIES
    ####################################
    if K_nt_dict['choice'] == 'IDEAL' or K_nt_dict['choice'] == 'IDEAL_LOAD':
       var_out_dict['Kv'][:,:,:]   = K_nt_dict['Kv']
       var_out_dict['Kt'][:,:,:]   = K_nt_dict['Kt']
       var_out_dict['ghat'][:,:,:] = K_nt_dict['ghat']
      
    if K_nt_dict['choice'] == 'KPP':
       var_out_dict['Kv'][0,:,:] = K_nt_dict['Kv'][0,:,:]
       var_out_dict['Kt'][0,:,:] = K_nt_dict['Kt'][0,:,:]



    
    ###################################################
    # FILL DICTIONARY VARIABLES WITH INITIAL CONDITION
    ###################################################
    keys_IC = ['u', 'v', 'phi_total', 'Kv', 'Kt', 'hbls', 'hbbl']    
    for ke in range(len(keys_IC)):
        rank_var = len(var_out_dict[keys_IC[ke]].shape)
        if rank_var == 3:
           var_out_dict[keys_IC[ke]][0,:,:] = IC_dict[keys_IC[ke]] 
        else:
            var_out_dict[keys_IC[ke]][0,:] = IC_dict[keys_IC[ke]] 



    ###################################################
    # BUOYANCY INITIAL CONDITION DEPENDENT ON SCHEME
    ##################################################
    if choices_dict['tstep_scheme_main'] == 'AB3_SI':   
       ##################################################################
        #PLACE IC for b at b_stagger n=3/2 (index 0 for 'b_stagger' array)
        '''
        'b'[n=1] (python, n=0) will get updated in time-stepping loop
         with simple interpolation
        '''
        ################################################################
        # THIS IS placed formally at n=3/2
        var_out_dict['b_stagger'][0,:,:] = IC_dict['b']

    if choices_dict['tstep_scheme_main'] == 'LF_SI':
       var_out_dict['b'][0,:,:] = IC_dict['b']
    ################################
    # CALCULATE w[n=0]
    ###############################
    var_out_dict['w'][0,:,:] = TTTW_func.calc_w(var_out_dict['v'][0,:,:],grids)



    ###########################################################
    ###########################################################

    #               TIME STEP 

    ##########################################################
    ##########################################################
    for n in range(nt-1):
        '''
        TIME-STAGGERING INDEXING


        u,v,w are time-stepped from n = 1 to n = 2

        b_stagger is time-stepped from n=3/2 to n=5/2

        this loop begins at n=1 (0 in python)
 
        and ends with u,v,w time-stepped to n=nt

        and b_stagger time-stepped to n=nt+1/2
        '''
        #########################################
        # DETERMINE SCHEME BASED ON TIME-STEP
        # AND SCHEME CHOSEN IN PARAMETERS
        #########################################
        if n==0:
           scheme_n = 'FE_SI'
        else:
           if choices_dict['tstep_scheme_main'] == 'AB3_SI':
              if n==1:
                scheme_n = 'AB2_SI'
              else:
                scheme_n = 'AB3_SI'
           if choices_dict['tstep_scheme_main'] =='LF_SI':
              scheme_n = 'LF_SI'

        #########################################
        # CALCULATE MAX COURANT # FOR PRINTING
        ########################################        
        courant_v = np.max(abs(var_out_dict['v'][n,:,:])) * dt / (grids['y_u_r'][1,0] - grids['y_u_r'][0,0])
        courant_w = np.max(abs(var_out_dict['w'][n,:,:])) * dt / (grids['z_u_r'][0,1] - grids['z_u_r'][0,0])

        #########################################
        # SOLVE FOR u,v,w at n+1
        #########################################
        print '##############################################'
        print ''
        print 'Total time-steps =', nt
        print 'MP_switch: ', choices_dict['MP_switch']
        if choices_dict['MP_switch']:
           print '# of processes: ', MP.cpu_count()-1
        print 'TIME-STEP SCHEME: ' + scheme_n
        print 'max(abs(v)) * dt / dy = ', courant_v
        print 'max(abs(w)) * dt / dz = ', courant_w

        print 'Solving for u,v,w at n =', (n+1)+1
        var_out_dict['u'][n+1,:,:], var_out_dict['v'][n+1,:,:], var_out_dict['w'][n+1,:,:] = solve_uvw(grids, var_out_dict, surf_flux_dict, choices_dict, constants_dict, n, dt, scheme_in=scheme_n)



        if choices_dict['tstep_scheme_main'] == 'AB3_SI':
           ##########################################
           # SOLVE FOR b_stagger n+3/2
           #########################################
           print 'Solving for b_stagger at n =', (n+1+1)+0.5
           var_out_dict['b_stagger'][n+1,:,:] = solve_b(grids, var_out_dict, surf_flux_dict, choices_dict, constants_dict,n, dt, scheme_in=scheme_n)
         
           ##############################################
           # INTERPOLATE b_stagger to u,v points for b
           ##############################################
           print 'Interpolating b'
           var_out_dict['b'][n,:,:] = 0.5 * (var_out_dict['b_stagger'][n+1,:,:] + var_out_dict['b_stagger'][n,:,:])
        
        if choices_dict['tstep_scheme_main'] == 'LF_SI':
           ##########################################
           # SOLVE FOR b  n+1
           #########################################
           print 'Solving for b at n =', (n+1)+1
           var_out_dict['b'][n+1,:,:] = solve_b(grids, var_out_dict, surf_flux_dict, choices_dict, constants_dict,n, dt, scheme_in=scheme_n)
         

        ##############################################
        # UPDATE TOTAL PRESSURE
        '''
        because of time-staggering
        and solveing for u,v at n=n+1 first,
        phi_total update uses buoyancy at
        n=n and ubar at n=n+1
        '''
        ##############################################
        print 'Updating phi_total to n=', (n+1)+1
        var_out_dict['phi_total'][n+1,:,:] = update_phi(n,var_out_dict, grids, constants_dict, surf_flux_dict,choices_dict) 
 
        ################################################
        # CALL KPP
        ###############################################
        if K_nt_dict['choice'] == 'KPP':
           var_out_dict['Kv'][n+1,:,:], var_out_dict['Kt'][n+1,:,:], var_out_dict['hbls'][n+1,:], var_out_dict['hbbl'][n+1,:], var_out_dict['ghat'][n+1,:,:] = tstep_kpp(n+1, var_out_dict, grids, surf_flux_dict, tstep_dict, constants_dict, choices_dict,tstep_scheme=scheme_n)

        

        print ''
        print '##############################################'        

    if choices_dict['tstep_scheme_main'] == 'AB3_SI':
       ###########################################
       # ASSIGN BUOYANCY FOR LAST INTEGER  TIME-STEP
       ##########################################
       var_out_dict['b'][n+1,:,:] = var_out_dict['b_stagger'][n+1,:,:]


    #################################################
    # RETURN VARIABLE DICTIONARY FOR NETCDF WRITING
    #################################################
    print ''
    print '##############################################################'
    print ''
    print ' MODEL RUN TIME: ' + str(time.time() - start_time) + ' seconds'
    print ''
    print '##############################################################'
    return var_out_dict
    ###################################################



#########################################################################################


                #########################################################################

                #               TIME STEP KPP FUNCTIONS

                #########################################################################
def tstep_kpp(n_time, var_out_dict, grids, surf_flux_dict, tstep_dict, constants_dict, choices_dict,tstep_scheme='AB3_SI'):
    '''
    FUNCTION THAT CALLS KPP MODULE
    AND CALCULATES Kv,Kt,hbls,hbbl,ghat
    for the next time-step
    '''
    ######################################
    # CALCULATE SURFACE HEAT FLUX FOR KPP
    #####################################
    srflx = surf_flux_dict['Bfsfc'][n_time,:] / (constants_dict['alpha'] * constants_dict['g'])
    

    ################################################
    # AVERAGE OLD GUESSES OF BL DEPTHS AND Kv, Kt
    ###############################################
    scheme_choices = ['AB3_SI', 'AB2_SI', 'LF_SI', 'FE_SI']
    #EXTRAPOLATION FUNCTION CHOICES
    extrap_funcs   = [extrap_AB3, extrap_AB2, extrap_FE, extrap_FE] 
    ind_s          = scheme_choices.index(tstep_scheme)
    ind_s = 3


    # n_time = n + 1 when called, so 'guess'es are at n_time-1
    Kv_old = extrap_funcs[ind_s](var_out_dict['Kv'],n_time-1)
    Kt_old = extrap_funcs[ind_s](var_out_dict['Kt'],n_time-1) 
    hbls_old = extrap_funcs[ind_s](var_out_dict['hbls'],n_time-1)
    hbbl_old = extrap_funcs[ind_s](var_out_dict['hbbl'],n_time-1)
 
    
    #####################################################
    # ASSIGN BUOYANCY KEY BASED ON TIME-STEPPING SCHEME
    ####################################################
    if tstep_scheme == 'LF_SI':
       b_in = var_out_dict['b'][n_time,:,:]
    else:
       b_in = var_out_dict['b'][n_time-1,:,:]
    #######################################
    # CREATE KPP OBJECT AND RUN KPP
    #######################################
    print 'Running KPP...'
    print 'n_time = ', n_time+1
    KPP_obj = KPP_mod.TTTW_KPP(b_in, var_out_dict['u'][n_time,:,:],var_out_dict['v'][n_time,:,:], hbls_old, hbbl_old, Kv_old, Kt_old, srflx, surf_flux_dict['sustr'][n_time,:], surf_flux_dict['svstr'][n_time,:],constants_dict['f'][:], grids, True, tstep_dict['dt'])

    KPP_obj.run_KPP()
    
    Kv_out   = KPP_obj.Kv_out
    Kt_out   = KPP_obj.Kt_out
    hbls_out = KPP_obj.hbls
    hbbl_out = KPP_obj.hbbl
    ghat_out = KPP_obj.ghat

    del KPP_obj

    return Kv_out, Kt_out, hbls_out, hbbl_out, ghat_out
    ###########################################################





  

                #########################################################################

                #               SOLVE FUNCTIONS: CALL ALL MATRIX SETUP AND INVERSION

                #########################################################################

def update_phi(n_time,var_out_dict, grids, constants_dict, surf_flux_dict,choices_dict):
    '''
    UPDATE PRESSURE FOR A NEW TIME-STEP WITH PRESSURE
    POISSON SOLVER
    '''
    ############################################# 
    # CALCULATE TERMS NEEDED FOR PRESSURE POISSON
    ###############################################
    ubar = np.mean(var_out_dict['u'][n_time+1,:,:],axis=1)
    phi_b = TTTW_func.calc_phi_baroclin(var_out_dict['b'][n_time,:,:], grids, constants_dict['rho0'],constants_dict['g']) 
    bustr, bvstr = TTTW_func.calc_bstress(var_out_dict['u'][n_time+1,:,:], var_out_dict['v'][n_time+1,:,:], constants_dict['Zob'], grids, constants_dict['rho0'],choices_dict['bottom_stress']) 

    #############################################
    return  TTTW_func.calc_phi_total(ubar, phi_b, surf_flux_dict['svstr'][n_time+1,:], bvstr,constants_dict['f'], grids, constants_dict['rho0'], BC_choice = 'doub_dirch')
    ###################################################################################

def solve_uvw(grids, var_out_dict, surf_flux_dict, choices_dict, constants_dict,n_time, dt, scheme_in='AB3_SI'):
    '''
    CALL FUCNTIONS THAT SETUP UKNOWN AND RHS MATRICES
    AND CALL FUNCTION TO INVERT TRIDGIONAL SYSTEM AND SOLVE

    CONVERT v to v-points and compute w

    returns u,v,w at respective points
    '''
    #####################################################
    # SET UP COEFFICIENT DICTIONAR FOR TRIDIAGNOAL SYSTEM
    ######################################################
    n_K = n_time - 1 
    if choices_dict['MP_switch']:
       coeff_dict = setup_A_coeff_dict_MP(grids, dt, var_out_dict['Kv'][n_K,:,:], tstep_scheme=scheme_in)
    else:
       coeff_dict = setup_A_coeff_dict(grids, dt, var_out_dict['Kv'][n_K,:,:],tstep_scheme=scheme_in)

    #####################################################
    # SET UP RHS ARRAY FOR TRIDIAGNOAL SYSTEM
    ######################################################
    R_full_uv = setup_RHS_momentum(grids, dt, n_time, var_out_dict, surf_flux_dict, choices_dict, constants_dict, tstep_scheme=scheme_in)  

    #######################################################
    #       SOLVE FOR u,v
    #######################################################
    [Ly,N] = grids['y_u_r'].shape
    u = np.zeros([Ly,N])
    v_upts = np.zeros([Ly,N])
    #u,v_upts = tridiag_invert_momentum(grids, coeff_dict, R_full_uv)
    if choices_dict['MP_switch']:
       '''
       IF MULTIPROCESSING USED
       '''
       pool = MP.Pool(processes=MP.cpu_count()-1)
       list_args = [(grids, coeff_dict, R_full_uv, j) for j in range(Ly)]
       uv_results = pool.map(MP_tridiag_momentum_wrapper, list_args)
       for j in range(Ly):
           u[j,:] = uv_results[j][0]
           v_upts[j,:] = uv_results[j][1]    
       pool.close()
       pool.join()
    else:
        for j in range(Ly):
            u[j,:], v_upts[j,:] = tridiag_invert_momentum_single_j(grids, coeff_dict, R_full_uv, j)
     
    ###############################################
    # CONVERT V TO v-points and solve for w
    #############################################
    v = TTTW_func.u2v(v_upts)
    w = TTTW_func.calc_w(v, grids)


    return u,v,w
    #################################################

def solve_b(grids, var_out_dict, surf_flux_dict, choices_dict, constants_dict,n_time, dt, scheme_in='AB3_SI'):
    '''
    CALL FUCNTIONS THAT SETUP UKNOWN AND RHS MATRICES
    AND CALL FUNCTION TO INVERT TRIDGIONAL SYSTEM AND SOLVE

    SOLVE for b_stagger at n+3/2 (if staggered time-grid used)

           OR
    SOVLE for b at n+1 (if non-staggered time-grid scheme used)
    '''
    #####################################################
    # SET UP COEFFICIENT DICTIONAR FOR TRIDIAGNOAL SYSTEM
    ######################################################
    n_K = n_time - 1
    if choices_dict['MP_switch']:
       coeff_dict = setup_A_coeff_dict_MP(grids, dt, var_out_dict['Kt'][n_K,:,:], tstep_scheme=scheme_in)
    else:
       coeff_dict = setup_A_coeff_dict(grids, dt, var_out_dict['Kt'][n_K,:,:],tstep_scheme=scheme_in)

    #####################################################
    # SET UP RHS ARRAY FOR TRIDIAGNOAL SYSTEM
    ######################################################
    R_full_b = setup_RHS_buoyancy(grids, dt, n_time, var_out_dict, surf_flux_dict, choices_dict, constants_dict, tstep_scheme=scheme_in)  

    #######################################################
    #       SOLVE FOR b or b_stagger
    ########################################################
    [Ly,N] = grids['y_u_r'].shape
    b_out = np.zeros([Ly,N])
    if choices_dict['MP_switch']:
       '''
       IF MULTIPROCESSING USED
       '''
       pool = MP.Pool(processes=MP.cpu_count() - 1)
       list_args = [(grids, coeff_dict, R_full_b, j) for j in range(Ly)]
       b_results = pool.map(MP_tridiag_b_wrapper, list_args)
       for j in range(Ly):
           b_out[j,:] = b_results[j]      
       pool.close()
       pool.join()
       return b_out
      
    else:
        for j in range(Ly):
            b_out[j,:] = tridiag_invert_buoyancy_single_j(grids, coeff_dict,R_full_b, j)
    return b_out
    #################################################




                #########################################################################

                #        TIME-STEPPING SCHEME EXTRAPOLATION FUNCTIONS

                #########################################################################

def extrap_AB3(var,n_time):
    '''
    FORM EXTRAPOLATION OF A VARIABILE
    BASED ON AB3 SCHEME
    
    var ---> 3D variable (t,y,z)
    '''
    alpha = 23./12.
    beta  = 4./3.
    gamma = 5./12.
    if len(var.shape)>2:
       return ( (alpha * var[n_time,:,:]) - (beta * var[n_time-1,:,:]) + (gamma * var[n_time-2,:,:]))
    else:
       return ( (alpha * var[n_time,:]) - (beta * var[n_time-1,:]) + (gamma * var[n_time-2,:]))

   ############################################################################################## 
 
def extrap_AB2(var,n_time):
    '''
    FORM EXTRAPOLATION OF A VARIABILE
    BASED ON AB2 SCHEME
    
    var ---> 3D variable (t,y,z)
    '''
    alpha = 3./2.
    beta  = 1./2.
    if len(var.shape)>2:
       return ( (alpha * var[n_time,:,:]) - (beta * var[n_time-1,:,:]) )
    else: 
       return ( (alpha * var[n_time,:]) - (beta * var[n_time-1,:]) )
    ############################################################################################## 

def extrap_FE(var,n_time):
    '''
    FORWARD EUELER (or leap-frog) USES variable at t=n, so just
    return that, but need funciton to maintain code structure 
    in setup_RHS_ functions 
    var ---> 3D variable (t,y,z)
    '''
    if len(var.shape)>2:
       return var[n_time,:,:]
    else:
       return var[n_time,:]
    ############################################################################################## 
 
                #########################################################################

                #        FUNCTIONS TO CALCULATE RHS TERMS IN momentum and buoyancy eqns

                #########################################################################


def horiz_diffusion(var, Dh, grids):
    '''
    CALCULATE HORIZONTAL DIFFUSION TERM
    FOR A VARIABLE (var) under
    a diffusion coefficient (Dh)

    DIFFUSION TERM CALCULATED USING CENTERED DIFFERENCING
    '''
    y_u_r = grids['y_u_r']
    y_v_r = grids['y_v_r']

    [Ly,N] = y_u_r.shape
    if var.shape[0] == Ly+1:
       var = TTTW_func.v2u(var)

    Dy = y_v_r[1:,:] - y_v_r[:-1,:]
    diff_flux = np.zeros([Ly,N])
    for j in range(1,Ly-1):
        diff_flux[j,:] = Dh * (1/Dy[j,:]**2) * (var[j+1,:] - 2*var[j,:] + var[j-1,:])

    return diff_flux
    ############################################################


def UV_rhs(u,v,phi_y,f_arr, grids):
    '''
    COMPUTE CORIOLIS AND PRESSURE GRADIENT
    RHS TERMS IN MOMENTUM EQNS USING FINITE-VOLUME
    DISCRETIZATION
    '''
    Hz = grids['z_u_w'][:,1:] - grids['z_u_w'][:,:-1]
    Dy = grids['y_v_r'][1:,:] - grids['y_v_r'][:-1,:]
    # GET DIMENSIONS
    [Ly,N] = u.shape
    ##########################
    # CALCULATE urhs, vrhs
    # in finite volume approach
    ##########################
    urhs = np.zeros([Ly,N])
    vrhs = np.zeros([Ly,N])
    for k in range(N):
        urhs[:,k] = (f_arr*TTTW_func.v2u(v[:,k])) * (Hz[:,k] * Dy[:,k])
        vrhs[:,k] = ( (-f_arr*u[:,k]) - phi_y[:,k]) * (Hz[:,k] * Dy[:,k])

    return urhs, vrhs
    #######################################################################

def b_adv_term(v,w,b,grid_dict):
    '''
    FINITE-VOLUME, FLUX-CONSERVATIVE
    CALCULATION OF BUOYANCY ADVECTION
    BY v and w
    b_adv = d/dy(vb) + d/dz(wb)
    '''
    [Ly,N] = b.shape

    ##################################
    # GET NECESSARY GRIDS
    # TO CALCULATE SPACING
    # FOR FLUXES THROUGH GRID CELL FACES
    #####################################
    z_v_w = grid_dict['z_v_w']
    y_v_w = grid_dict['y_v_w']
    dz = z_v_w[:,1:] - z_v_w[:,:-1]
    dy = y_v_w[1:,:] - y_v_w[:-1,:]
    b_adv = np.zeros([Ly,N])

    horiz_flux = np.empty([Ly,N])
    vert_flux = np.empty([Ly,N])
    ##################################
    # INTERPOLATE b to v and w grids
    ################################
    b_v_r = TTTW_func.u2v(b)
    b_u_w = TTTW_func.rho2w(b)
    horiz_flux = (b_v_r[1:,:] * v[1:,:] * dz[1:,:]) - (b_v_r[:-1,:] *v[:-1,:] * dz[:-1,:])
    vert_flux  = (b_u_w[:,1:] * w[:,1:] * dy[:,1:]) - (b_u_w[:,:-1] * w[:,:-1] * dy[:,:-1])
    '''
    for j in range(Ly):
        for k in range(N):
            horiz_flux[j,k] = (b_v_r[j+1,k] * v[j+1,k] * dz[j+1,k]) - (b_v_r[j,k] *v[j,k] * dz[j,k])
            vert_flux[j,k]  = (b_u_w[j,k+1] * w[j,k+1] * dy[j,k+1]) - (b_u_w[j,k] * w[j,k] * dy[j,k])
    '''
    return vert_flux + horiz_flux
    #######################################################################


def nonlocal_KPP_rhs(ghat,Bfsfc,grid_dict):
    '''
    CALCULATE NON-LOCAL VERTICAL MIXING OF TRACER
    (BUOYANCY)  TERM OBTAINED FROM ghat in KPP
 
    dNL_dz = d/dz(ghat * Bfsfc)

    '''
    z_u_w = grid_dict['z_u_w']
    [Ly,N] = grid_dict['z_u_r'].shape
    Hz = z_u_w[:,1:] - z_u_w[:,:-1]

    dNL_dz = np.zeros([Ly,N])
    cff = np.zeros([Ly,N+1])
    for j in range(Ly):  
        cff[j,:] = ghat[j,:] * Bfsfc[j]
    
    dNL_dz = (cff[:,1:] - cff[:,:-1]) / Hz
    return dNL_dz
    ###############################################################


                #########################################################################

                #        FUNCTIONS TO SETUP MATRICES FOR TRIDIAGONAL INVERSION

                #########################################################################

def setup_RHS_buoyancy(grids, dt,n_time, var_out_dict, surf_flux_dict, choices_dict, constants_dict, tstep_scheme='AB3_SI'): 
    '''
    CREATE ARRAY OF RHS TERMS FOR BUOYANCY EQUATION
    TRIDIAGONAL MATRIX SYSTEM

    n_time ---> time index [n] at point in time-stepping
    
    RHS FOR buoyancy has general form
         Dy * Hz * b_pre - (dt_amp * b_adv[n_SCHEME,:,:])

        WHERE n_pre is deteremined by time-stepping scheme (e.g., n_pre = n_time-1 or n_pre = n_time)
        AND     n_SCHEME is not exactly time-index, but can imply extrapolated values IF AB2, AB3 schemes are used
        AND dt_amp is factor of time-step to be used depending on time-stepping scheme (e.g., dt_amp = 2 * dt for LF_SI scheme) 
    '''
    ##########################################
    # LISTS THAT DETERMINE FORM OF TERMS
    #########################################
    # TYPES OF SCHEMES TO USE
    scheme_choices = ['AB3_SI', 'AB2_SI', 'LF_SI', 'FE_SI']
    #EXTRAPOLATION FUNCTION CHOICES
    extrap_funcs   = [extrap_AB3, extrap_AB2, extrap_FE, extrap_FE] 
    #AMPLITUDE FACTOR FOR TIME-DERIVATIVE
    dt_amp_list    = [dt, dt, 2*dt,dt]
    #TIME-INDEX FOR KNOWN TERM IN TIME-DERIVATIVE
    n_pre_list = [n_time, n_time, n_time-1,n_time] 
    #TIME INDEX FOR ADVECTIVE VELOCITIES
    adv_n_list = [n_time+1, n_time+1,n_time,n_time+1]
    #TIME INDEX FOR SURFACE BUOYANCY FLUX
    n_surf_list = [n_time, n_time, n_time,n_time]
 
    ##########################
    #INDEX FOR SCHEME CHOSEN
    ########################
    ind_s          = scheme_choices.index(tstep_scheme)
 

    ######################################################
    #SET buoyancy variable key (stagger or non-staggered) 
    #based on time-stepping scheme chosen in main
    #####################################################
    if choices_dict['tstep_scheme_main'] == 'AB3_SI':
       b_key = 'b_stagger'
    if choices_dict['tstep_scheme_main'] == 'LF_SI':
       b_key = 'b'
 
    #########################################
    # CALCULATE ADVECTIVE TERM FOR BUOYANCY 
    #########################################
    [Ly,N] = grids['y_u_r'].shape
    b_adv_FV = np.zeros([Ly,N])
    #IF BUOYANCY ADVECTION IS SWITCHED ON, CALCULATE ADVECTIVE FLUXES
    if choices_dict['advect_bool']:
       n_adv = adv_n_list[ind_s]
       b_extrap   = extrap_funcs[ind_s](var_out_dict[b_key],n_time)
       b_adv_FV = b_adv_term(var_out_dict['v'][n_adv,:,:], var_out_dict['w'][n_adv,:,:], b_extrap, grids)

    ##############################################
    # SET FORCING TIME FOR SURFACE BUOYANCY FLUX
    ###############################################
    n_surf = n_surf_list[ind_s]
    #srflx = surf_flux_dict['Bfsfc'][n_surf,:] / (constants_dict['g'] * constants_dict['alpha'])

    ##################################
    # SET dt according to scheme
    ##################################
    dt_amp = dt_amp_list[ind_s]
 
    #########################################
    # SETUP b_pre TERM
    ##########################################
    #this list matches up with scheme_choices above
    n_pre = n_pre_list[ind_s]
   
    Hz = grids['z_u_w'][:,1:] - grids['z_u_w'][:,:-1]
    Dy = grids['y_v_r'][1:,:] - grids['y_v_r'][:-1,:]
    
    #CREATE FINITE-VOLUME KNOWN-TERMS IN TIME-DERIVATIVE
    b_pre_FV = Hz * Dy * var_out_dict[b_key][n_pre,:,:]


    ########################################
    # CALCULATE NON-LOCAL VERTICAL MIXING
    ########################################
    dNL_dz = nonlocal_KPP_rhs(var_out_dict['ghat'][n_surf,:,:],surf_flux_dict['Bfsfc'][n_surf,:],grids)


    #############################################
    # CREATE RHS MATRIX
    # TO BE USED IN VERTICAL TRIDIAGONAL SYSTEM     
    #############################################
    R_full = np.zeros([Ly,N])    
    ###########################
    # FILL BOTTOM
    ############################ 
    R_full[:,0] = b_pre_FV[:,0] - (dt_amp * b_adv_FV[:,0]) - (dt_amp * Dy[:,0] * dNL_dz[:,0]) 
    ###########################
    # FILL INTERIOR
    ############################ 
    R_full[:,1:N-1] = b_pre_FV[:,1:N-1] - (dt_amp*b_adv_FV[:,1:N-1]) -  (dt_amp * Dy[:,1:N-1] * dNL_dz[:,1:N-1])
    ########################
    # FILL TOP
    ######################### 
    R_full[:,-1] = b_pre_FV[:,-1] - (dt_amp * b_adv_FV[:,-1]) + (dt_amp * surf_flux_dict['Bfsfc'][n_surf,:] * Dy[:,-1]) -  (dt_amp * Dy[:,-1] * dNL_dz[:,-1]) 
 
    ######################################
    # HORIZONTAL DIFFUSION
    ######################################
    if choices_dict['horiz_diff']:    
       b_extrap = extrap_funcs[ind_s](var_out_dict[b_key],n_time)        
       b_diff_horiz = horiz_diffusion(b_extrap,constants_dict['Dh'], grids)
      
       R_full = R_full + (dt_amp * Hz * b_diff_horiz) 

    return R_full
    #############################################################################


def setup_RHS_momentum(grids, dt,n_time, var_out_dict, surf_flux_dict, choices_dict, constants_dict, tstep_scheme='AB3_SI'): 
    '''
    CREATE ARRAY OF RHS TERMS FOR MOMENTUM EQUATION
    TRIDIAGONAL MATRIX SYSTEM

    n_time ---> time index [n] at point in time-stepping


    RHS FOR MOMENTUM HAS GENEAL FORM:
    X-COMP:   Dy[j,k] * Hz[j,k] *  u[n_pre,j,k] + dt_amp * f[j]*v[n_SCHEME,j,k]

    Y-COMP:   Dy[j,k] * Hz[j,k] * v[n_pre,j,k] + dt_amp * (-f[j] * u[n_SCHME,j,k] - phi_y[n_SCHEME,j,k])

          ====> A + B, A is other side of d/dt derivative and B is other terms in momentum eqn (PGF, coriolis, horizontal diffusion)
        WHERE n_pre is deteremined by time-stepping scheme (e.g., n_pre = n_time-1 or n_pre = n_time)
        AND     n_SCHEME is not exactly time-index, but can imply extrapolated values IF AB2, AB3 schemes are used
        AND dt_amp is factor of time-step to be used depending on time-stepping scheme (e.g., dt_amp = 2 * dt for LF_SI scheme) 
    '''
    ##########################################
    # LISTS THAT DETERMINE FORM OF TERMS
    #########################################
    # TYPES OF SCHEMES TO USE
    scheme_choices = ['AB3_SI', 'AB2_SI', 'LF_SI','FE_SI']
    #EXTRAPOLATION FUNCTION CHOICES
    extrap_funcs   = [extrap_AB3, extrap_AB2, extrap_FE, extrap_FE] 
    #AMPLITUDE FACTOR FOR TIME-DERIVATIVE
    dt_amp_list    = [dt, dt, 2*dt, dt]
    #TIME-INDEX FOR KNOWN TERM IN TIME-DERIVATIVE
    n_pre_list = [n_time, n_time, n_time-1,n_time] 
    #TIME-INDEX FOR STRESSES
    n_stress_list   = [n_time, n_time, n_time,n_time]

    ##########################
    #INDEX FOR SCHEME CHOSEN
    ########################
    ind_s          = scheme_choices.index(tstep_scheme)

    [Ly,N] = grids['y_u_r'].shape
 
    #########################################
    # SET UP u,v,phi w/in B-term
    # depending on time-stepping scheme
    #########################################
    u_extrap   = extrap_funcs[ind_s](var_out_dict['u'],n_time)
    v_extrap   = extrap_funcs[ind_s](var_out_dict['v'],n_time)
    phi_extrap = extrap_funcs[ind_s](var_out_dict['phi_total'],n_time)
   
    Urhs_FV, Vrhs_FV = UV_rhs(u_extrap, v_extrap, TTTW_func.horiz_grad(phi_extrap, grids['y_u_r']),constants_dict['f'],grids) 
    dt_amp = dt_amp_list[ind_s]

    #########################################
    # SETUP A TERM
    ##########################################
    #this list matches up with scheme_choices above
    n_pre = n_pre_list[ind_s]
   
    Hz = grids['z_u_w'][:,1:] - grids['z_u_w'][:,:-1]
    Dy = grids['y_v_r'][1:,:] - grids['y_v_r'][:-1,:]
    
    #CREATE FINITE-VOLUME KNOWN-TERMS IN TIME-DERIVATIVE
    u_pre_FV = Hz * Dy * var_out_dict['u'][n_pre,:,:]
    v_pre_FV = Hz * Dy * TTTW_func.v2u(var_out_dict['v'][n_pre,:,:])   

    #####################################
    #CALCULATE BOTTOM STRESSES AT n_time 
    ######################################
    n_stress = n_stress_list[ind_s]
    bustr, bvstr = TTTW_func.calc_bstress(var_out_dict['u'][n_stress,:,:], var_out_dict['v'][n_stress,:,:], constants_dict['Zob'], grids, constants_dict['rho0'],choices_dict['bottom_stress']) 

    #############################################
    # LOOP ALONG j and CREATE RHS MATRIX
    # TO BE USED IN VERTICAL TRIDIAGONAL SYSTEM     
    #############################################
    R_full = np.zeros([Ly,N*2])    


    ########################
    # FILL BOTTOM
    ########################
    R_full[:,0] = u_pre_FV[:,0] +  (dt_amp * Urhs_FV[:,0]) - (Dy[:,0] * dt_amp * bustr)
    R_full[:,1] = v_pre_FV[:,0] +  (dt_amp * Vrhs_FV[:,0]) - (Dy[:,0] * dt_amp * TTTW_func.v2u(bvstr))
         
    ########################
    # FILL INTERIOR
    ########################
    for k in range(1,N-1):
        idx = 2*k
        R_full[:,idx]   = u_pre_FV[:,k] + (dt_amp * Urhs_FV[:,k])
        R_full[:,idx+1] = v_pre_FV[:,k] + (dt_amp * Vrhs_FV[:,k]) 

    #######################
    # FILL TOP
    #######################
    R_full[:,-2] = u_pre_FV[:,-1] + (dt_amp * Urhs_FV[:,-1]) + (Dy[:,-1]*dt_amp * surf_flux_dict['sustr'][n_stress,:])
    R_full[:,-1] = v_pre_FV[:,-1] + (dt_amp * Vrhs_FV[:,-1]) + (Dy[:,-1]*dt_amp * TTTW_func.v2u(surf_flux_dict['svstr'][n_stress,:]))

 
    ######################################
    # HORIZONTAL DIFFUSION
    ######################################
    if choices_dict['horiz_diff']:    
       u_diff_horiz = horiz_diffusion(u_extrap,constants_dict['Dh'], grids)
       v_diff_horiz = horiz_diffusion(v_extrap,constants_dict['Dh'], grids)
      
       R_full[:,0] = R_full[:,0] + (dt_amp * Hz[:,0] * u_diff_horiz[:,0]) 
       R_full[:,1] = R_full[:,1] + (dt_amp * Hz[:,0] * v_diff_horiz[:,0]) 
       for k in range(1,N-1):
           idx = 2*k
           R_full[:,idx]   = R_full[:,idx] + (dt_amp * Hz[:,k] * u_diff_horiz[:,k])
           R_full[:,idx+1] = R_full[:,idx+1] + (dt_amp * Hz[:,k] * v_diff_horiz[:,k])

       R_full[:,-2] = R_full[:,-2] + (dt_amp * Hz[:,-1] * u_diff_horiz[:,-1]) 
       R_full[:,-1] = R_full[:,-1] + (dt_amp * Hz[:,-1] * v_diff_horiz[:,-1]) 
     


    return R_full
    ################################################################################


def setup_A_coeff_dict_MP(grids, dt, A, tstep_scheme='AB3_SI'):
   '''
   SETUP DICTIONARY OF COEFFCIENT VALUES USING
   MULTIPROCESSING POOL FUNCITONALITY TO SPEED UP
   MODEL RUN TIME
   '''
   [Ly,N] = grids['y_u_r'].shape
   ######################################
   # SETUP DICTIONARY 
   ######################################
   coeff_dict = {} 
   new_keys = ['D_1', 'D_2', 'D_k_m_1', 'D_k', 'D_k_p_1', 'D_N_m_1', 'D_N']
   key_shapes = [[Ly], [Ly], [Ly,N], [Ly,N], [Ly,N], [Ly], [Ly]]
   for ke in range(len(new_keys)):
       coeff_dict[new_keys[ke]] = np.zeros(key_shapes[ke])

   pool = MP.Pool(processes=MP.cpu_count()-1)
   list_args = [(grids, dt, A, j, tstep_scheme) for j in range(Ly)]
   A_results = pool.map(MP_setup_A_coeff_dict_wrapper, list_args)
   for j in range(Ly):
       coeff_dict['D_1'][j] = A_results[j][0]
       coeff_dict['D_2'][j] = A_results[j][1]
       coeff_dict['D_k_m_1'][j,:] = A_results[j][2]
       coeff_dict['D_k'][j,:] = A_results[j][3]
       coeff_dict['D_k_p_1'][j,:] = A_results[j][4]
       coeff_dict['D_N_m_1'][j] = A_results[j][5]
       coeff_dict['D_N'][j] = A_results[j][6]

   pool.close()
   pool.join()

   return coeff_dict
   ######################################################



def MP_setup_A_coeff_dict_wrapper(args):
    '''
    WRAPPER FUNCTION FOR COEFFICIENT DICTIONARY SETUP
    '''
    return setup_A_coeff_dict_single_j(*args)
    #################################################
 

def setup_A_coeff_dict_single_j(grids, dt, A, j,tstep_scheme='AB3_SI'):
    '''
    SET UP MATRIX OF UKNOWN COEFFICEINTS BASED ON 
    THE TYPE OF TIME-STEPPING SCHEME USED
   
    A --> vertical mixing [Ly,N+1], can be either Kv or Kt depending
          on if the function is being called for buoyancy eqn solution (Kt)
          or momentum eqn solutino (Kv)


    single_j ---> j-point to return  
    returns coeff_dict which is used in tridiagonal solver functions
    '''
    [Ly,N] = grids['y_u_r'].shape
    
    ################################
    # SPATIAL DIFFERENCES
    '''
    dz array is at w-levels
    dz should be positive everywhere
    dz shape is [Ly,N-1]

    Hz array  is at rho-levels
    Hz = vertical grid box height

    Dy array is at u-points, rho levels
    Dy = horizontal spacing btwn v-points

    '''
    ###############################
    dz = grids['z_u_r'][:,1:] - grids['z_u_r'][:,:-1]
    Hz = grids['z_u_w'][:,1:] - grids['z_u_w'][:,:-1]
    Dy = grids['y_v_r'][1:,:] - grids['y_v_r'][:-1,:]


    ######################################
    # SETUP ARRAYS TO RETURN 
    ######################################
    D_k_m_1 = np.zeros(N)
    D_k     = np.zeros(N)
    D_k_p_1 = np.zeros(N)

    ################################
    # SET delta_t AMPLFICATION FACTOR BASED
    # ON TIME-STEPPING SCHEME USED
    ################################
    scheme_choices = ['AB3_SI', 'AB2_SI', 'LF_SI','FE_SI']
    dt_amp_list    = [dt, dt, 2*dt, dt] 
    ##########################
    #INDEX FOR SCHEME CHOSEN
    ########################
    ind_s          = scheme_choices.index(tstep_scheme)
   
    dt_amp = dt_amp_list[ind_s]

    #######################################
    # FILL COEFFICIENTS 
    ####################################### 
    D_2  = Dy[j,0] * (dt_amp * (A[j,1] / dz[j,0]))
    D_1  = ( Dy[j,0] * Hz[j,0] )  + D_2 

    for k in range(1,N-1):
        D_k_m_1[k] = Dy[j,k] * (dt_amp * (A[j,k] / dz[j,k-1])) 
        D_k_p_1[k] = Dy[j,k] * (dt_amp * (A[j,k+1] / dz[j,k]))
        D_k[k] = ( Dy[j,k] * Hz[j,k] ) + D_k_m_1[k] + D_k_p_1[k]

    D_N_m_1 = Dy[j,-1] * (dt_amp * (A[j,N-1] / dz[j,-1]))
    D_N     = ( Dy[j,-1] * Hz[j,-1] ) + D_N_m_1
    #############################################################################

    return D_1, D_2, D_k_m_1, D_k, D_k_p_1, D_N_m_1, D_N
    ############################################################








def setup_A_coeff_dict(grids, dt, A, tstep_scheme='AB3_SI'):
    '''
    SET UP MATRIX OF UKNOWN COEFFICEINTS BASED ON 
    THE TYPE OF TIME-STEPPING SCHEME USED
   
    A --> vertical mixing [Ly,N+1], can be either Kv or Kt depending
          on if the function is being called for buoyancy eqn solution (Kt)
          or momentum eqn solutino (Kv)
 
    returns coeff_dict which is used in tridiagonal solver functions
    '''
    [Ly,N] = grids['y_u_r'].shape
    
    ################################
    # SPATIAL DIFFERENCES
    '''
    dz array is at w-levels
    dz should be positive everywhere
    dz shape is [Ly,N-1]

    Hz array  is at rho-levels
    Hz = vertical grid box height

    Dy array is at u-points, rho levels
    Dy = horizontal spacing btwn v-points

    '''
    ###############################
    dz = grids['z_u_r'][:,1:] - grids['z_u_r'][:,:-1]
    Hz = grids['z_u_w'][:,1:] - grids['z_u_w'][:,:-1]
    Dy = grids['y_v_r'][1:,:] - grids['y_v_r'][:-1,:]


    ######################################
    # SETUP DICTIONARY 
    ######################################
    coeff_dict = {} 
    new_keys = ['D_1', 'D_2', 'D_k_m_1', 'D_k', 'D_k_p_1', 'D_N_m_1', 'D_N']
    key_shapes = [[Ly], [Ly], [Ly,N], [Ly,N], [Ly,N], [Ly], [Ly]]
    for ke in range(len(new_keys)):
        coeff_dict[new_keys[ke]] = np.zeros(key_shapes[ke])


    ################################
    # SET delta_t AMPLFICATION FACTOR BASED
    # ON TIME-STEPPING SCHEME USED
    ################################
    scheme_choices = ['AB3_SI', 'AB2_SI', 'LF_SI','FE_SI']
    dt_amp_list    = [dt, dt, 2*dt, dt] 
    ##########################
    #INDEX FOR SCHEME CHOSEN
    ########################
    ind_s          = scheme_choices.index(tstep_scheme)
   
    dt_amp = dt_amp_list[ind_s]

    #######################################
    # FILL COEFFICIENTS 
    ####################################### 
    for j in range(Ly):
        coeff_dict['D_2'][j] = Dy[j,0] * (dt_amp * (A[j,1] / dz[j,0]))
        coeff_dict['D_1'][j] = ( Dy[j,0] * Hz[j,0] )  + coeff_dict['D_2'][j] 

        for k in range(1,N-1):
            coeff_dict['D_k_m_1'][j,k] = Dy[j,k] * (dt_amp * (A[j,k] / dz[j,k-1])) 
            coeff_dict['D_k_p_1'][j,k] = Dy[j,k] * (dt_amp * (A[j,k+1] / dz[j,k]))
            coeff_dict['D_k'][j,k] = ( Dy[j,k] * Hz[j,k] ) + coeff_dict['D_k_m_1'][j,k] + coeff_dict['D_k_p_1'][j,k]

        coeff_dict['D_N_m_1'][j] = Dy[j,-1] * (dt_amp * (A[j,N-1] / dz[j,-1]))
        coeff_dict['D_N'][j]     = ( Dy[j,-1] * Hz[j,-1] ) + coeff_dict['D_N_m_1'][j]
        #############################################################################

    return coeff_dict
    ############################################################




                #########################################################################

                #               TRIDIAGONAL SOLVER FUNCTIONS

                #########################################################################
def MP_tridiag_b_wrapper(args):
    '''
    WRAPPER FUNCTION FOR BUOYANCY TRIDIAGONAL INVERSION
    '''
    return tridiag_invert_buoyancy_single_j(*args)
    #################################################
    
def MP_tridiag_momentum_wrapper(args):
    '''
    WRAPPER FUNCTION FOR MOMENTUM TRIDIAGONAL INVERSION
    '''
    return tridiag_invert_momentum_single_j(*args)
    #################################################
 


def tridiag_invert_buoyancy_single_j(grids, coeff_dict, R_full,j):
    '''
    FUNCTION THAT INVERTS VERTICAL TRIDIAGONAL
    SYSTEM FOR BUOYANCY EQUATION

    j0 --> does vertical tridiag inversion for this j-values
               to be used for multiprocessing fuctionality to speed up
               run-time

    INPUTS:
    grids      ---> dictionary with grid values
    coeff_dict --> dictionary containing coefficients for uknonwns 
               coeff_dict['D_N'] ===> [Ly] shape (has coeffcient stored at every j-value)
    R_full          ---> array of knowns (RHS values, [Ly,N] shape)
    j0, j1 ---> range of j-values to loop over
    '''
    
    [Ly,N] = grids['y_u_r'].shape

    #############################
    # FORM COEFFICIENT MATRIX
    #############################
    new = np.zeros(N)
    b_out_j = copy(new)
    ndim = N
    A = lil_matrix((ndim, ndim))
     
    ##############################
    # FILL BOTTOM
    ###############################
    idx = 0
    A[idx,idx]   = coeff_dict['D_1'][j]
    A[idx,idx+1] = -coeff_dict['D_2'][j]
       
    ##############################
    # FILL INTERIOR
    ###############################
    for k in range(1,N-1):
        idx = k
        A[idx,idx-1] = -coeff_dict['D_k_m_1'][j,k]
        A[idx,idx]   = coeff_dict['D_k'][j,k]
        A[idx,idx+1] = -coeff_dict['D_k_p_1'][j,k] 
   
    ##############################
    # FILL SURFACE
    ################################
    idx = N-1
    A[idx,idx-1] = -coeff_dict['D_N_m_1'][j]
    A[idx,idx]   = coeff_dict['D_N'][j]

    #####################################
    # SOLVE 
    #####################################
    return spsolve(A.tocsr() , R_full[j])
        
    ##############################################


def tridiag_invert_momentum_single_j(grids, coeff_dict, R_full,j):
    '''
    FUNCTION THAT INVERTS VERTICAL TRIDIAGONAL
    SYSTEM FOR COUPLE MOMENTUM EQUATIONS

    single_j ---> can be used for multiprocessing

    INPUTS:
    grids      ---> dictionary with grid values
    coeff_dict --> dictionary containing coefficients for uknonwns 
               coeff_dict['D_N'] ===> [Ly] shape (has coeffcient stored at every j-value)
               coeff_dict['D_k'] ==> [Ly,N] shape has coefficients at j,k indices (interior points)
              
    R_full          ---> array of knowns (RHS values, [Ly,N] shape)
    '''
    
    [Ly,N] = grids['y_u_r'].shape

    #############################
    # FORM COEFFICIENT MATRIX
    #############################
    new = np.zeros([N])
    u_out, v_out = copy(new), copy(new)
    ndim = 2*N
    kv = 2

    A = lil_matrix((ndim, ndim))
       
    ##############################
    # FILL BOTTOM
    ###############################
    idx = 0
    A[idx,idx]          = coeff_dict['D_1'][j]
    A[idx,idx+kv]       = -coeff_dict['D_2'][j]
    A[idx+1, idx+1]     = coeff_dict['D_1'][j]
    A[idx+1,idx+1+kv]   = -coeff_dict['D_2'][j]
        
    ##############################
    # FILL INTERIOR
    ###############################
    for k in range(1,N-1):
        idx = kv*k
        A[idx,idx-kv] = -coeff_dict['D_k_m_1'][j,k] 
        A[idx,idx]   = coeff_dict['D_k'][j,k]
        A[idx,idx+kv] = -coeff_dict['D_k_p_1'][j,k] 
        A[idx+1,idx+1-kv] = -coeff_dict['D_k_m_1'][j,k] 
        A[idx+1,idx+1]   = coeff_dict['D_k'][j,k]
        A[idx+1,idx+1+kv] = -coeff_dict['D_k_p_1'][j,k] 
 
    ################################
    # FILL SURFACE
    ################################
    idx =(2*N)-kv
    A[idx,idx-kv] = -coeff_dict['D_N_m_1'][j]
    A[idx,idx]   = coeff_dict['D_N'][j]
    A[idx+1,idx+1-kv] = -coeff_dict['D_N_m_1'][j]
    A[idx+1,idx+1]   = coeff_dict['D_N'][j]


    #####################################
    # SOLVE 
    ####################################
    X = spsolve(A.tocsr(), R_full[j])
        
    ##################################
    # REORDER RESULTS IN [j,k]
    #################################
    for k in range(N):
        idx = 2*k
        u_out[k] = X[idx]
        v_out[k] = X[idx+1]        
  
    return u_out, v_out
    ##############################################


    
def tridiag_invert_buoyancy(grids, coeff_dict, R_full):
    '''
    FUNCTION THAT INVERTS VERTICAL TRIDIAGONAL
    SYSTEM FOR BUOYANCY EQUATION


    INPUTS:
    grids      ---> dictionary with grid values
    coeff_dict --> dictionary containing coefficients for uknonwns 
               coeff_dict['D_N'] ===> [Ly] shape (has coeffcient stored at every j-value)
    R_full          ---> array of knowns (RHS values, [Ly,N] shape)
    '''
    
    [Ly,N] = grids['y_u_r'].shape

    #############################
    # FORM COEFFICIENT MATRIX
    #############################
    new = np.zeros([Ly,N])
    b_out = copy(new)
    ndim = N
    for j in range(Ly): 
        A = lil_matrix((ndim, ndim))
       
        ##############################
        # FILL BOTTOM
        ###############################
        idx = 0
        A[idx,idx]   = coeff_dict['D_1'][j]
        A[idx,idx+1] = -coeff_dict['D_2'][j]
       
        ##############################
        # FILL INTERIOR
        ###############################
        for k in range(1,N-1):
            idx = k
            A[idx,idx-1] = -coeff_dict['D_k_m_1'][j,k]
            A[idx,idx]   = coeff_dict['D_k'][j,k]
            A[idx,idx+1] = -coeff_dict['D_k_p_1'][j,k] 
   
        ################################
        # FILL SURFACE
        ################################
        idx = N-1
        A[idx,idx-1] = -coeff_dict['D_N_m_1'][j]
        A[idx,idx]   = coeff_dict['D_N'][j]

        #####################################
        # SOLVE 
        #####################################
        X = spsolve(A.tocsr() , R_full[j])
        
        ##################################
        # REORDER RESULTS IN [j,k]
        #################################
        for k in range(N):
            b_out[j,k]=X[k]
  
    return b_out
    ##############################################



def tridiag_invert_momentum(grids, coeff_dict, R_full):
    '''
    FUNCTION THAT INVERTS VERTICAL TRIDIAGONAL
    SYSTEM FOR COUPLE MOMENTUM EQUATIONS


    INPUTS:
    grids      ---> dictionary with grid values
    coeff_dict --> dictionary containing coefficients for uknonwns 
               coeff_dict['D_N'] ===> [Ly] shape (has coeffcient stored at every j-value)
               coeff_dict['D_k'] ==> [Ly,N] shape has coefficients at j,k indices (interior points)
              
    R_full          ---> array of knowns (RHS values, [Ly,N] shape)
    '''
    
    [Ly,N] = grids['y_u_r'].shape

    #############################
    # FORM COEFFICIENT MATRIX
    #############################
    new = np.zeros([Ly,N])
    u_out, v_out = copy(new), copy(new)
    ndim = 2*N
    kv = 2
    for j in range(Ly): 
        A = lil_matrix((ndim, ndim))
       
        ##############################
        # FILL BOTTOM
        ###############################
        idx = 0
        A[idx,idx]          = coeff_dict['D_1'][j]
        A[idx,idx+kv]       = -coeff_dict['D_2'][j]
        A[idx+1, idx+1]     = coeff_dict['D_1'][j]
        A[idx+1,idx+1+kv]   = -coeff_dict['D_2'][j]
        
        ##############################
        # FILL INTERIOR
        ###############################
        for k in range(1,N-1):
            idx = kv*k
            A[idx,idx-kv] = -coeff_dict['D_k_m_1'][j,k] 
            A[idx,idx]   = coeff_dict['D_k'][j,k]
            A[idx,idx+kv] = -coeff_dict['D_k_p_1'][j,k] 
            A[idx+1,idx+1-kv] = -coeff_dict['D_k_m_1'][j,k] 
            A[idx+1,idx+1]   = coeff_dict['D_k'][j,k]
            A[idx+1,idx+1+kv] = -coeff_dict['D_k_p_1'][j,k] 
   
        ################################
        # FILL SURFACE
        ################################
        idx =(2*N)-kv
        A[idx,idx-kv] = -coeff_dict['D_N_m_1'][j]
        A[idx,idx]   = coeff_dict['D_N'][j]
        A[idx+1,idx+1-kv] = -coeff_dict['D_N_m_1'][j]
        A[idx+1,idx+1]   = coeff_dict['D_N'][j]


        #####################################
        # SOLVE 
        #####################################
        X = spsolve(A.tocsr(), R_full[j])
        
        ##################################
        # REORDER RESULTS IN [j,k]
        #################################
        for k in range(N):
            idx = 2*k
            u_out[j,k] = X[idx]
            v_out[j,k] = X[idx+1]        
  
    return u_out, v_out
    ##############################################


