######################################
__title__          = "TTTW_set_IC.py"
__author__         = "Daniel Dauhajre"
__date__           = "September 2017"
__email__          = "ddauhajre@atmos.ucla.edu"
__python_version__ = "2.7.9"

'''

OBTAIN INITIAL CONDITIONS
VIA KPP AND STEADY STATE
TTW

ITERATE WITH TTW AND KPP
UNTIL THERE IS CONVERGENCE
OF KPP MIXING AND TTW VELOCITIES

OR 

CREATE INITIAL CONDITONS AS 
PRESCRIBED BY USER

'''
######################################

###################################
#	IMPORT LIBRARIES
###################################
import os
import sys
import numpy as np
import scipy as sp
from pylab import *
import TTTW_funcs as TTTW_func
import TTTW_kpp as KPP
from netCDF4 import Dataset
import pickle as pickle
##################################


# 				COPY OF MAIN() TO GET IC's for KPP TESTING #


#################################
# 	SET CONSTANTS
################################
code_path = './src_code/'
execfile(code_path+'params.py')

print '		################################################'
print '             		RUN ID: ' + run_ID  
print '		################################################'




#################################################################################################

#                                   SETUP GRID AND INITIAL CONDITONS

##################################################################################################


################################
# CREATE GRID
################################
######################################
# Ly --> number of horizontal u-points
# Ly-1 --> number of horizontal v-points
# N ---> number of vertical rho-points
# N+1 --> number of vertical w-points
#######################################
grids,Ly,N = TTTW_func.make_grid(Ly_m,Lz_m,dy_m,dz_m)
# PLOT GRID IN RHO AND U-point coordinates
y_u_r = grids['y_u_r']
z_u_r = grids['z_u_r']
y_v_r = grids['y_v_r']
z_u_w = grids['z_u_w']
y_u_w = grids['y_u_w']
z_v_w = grids['z_v_w']
y_v_w = grids['y_v_w']
grid_ext = [y_u_r[0,0], y_u_r[-1,-1], 0,z_u_r.shape[1]]


#################################################
# SET CORIOLIS PARMAMETER FOR TTW CALCULATION
################################################
f_arr = np.zeros([Ly])
f_arr[:] = coriolis_param


####################################
# CREATE AND SET INITIAL CONDITIONS
'''
Depending on IC case, set b(y,z),K(y,z),gamma(y,z)
u(y,z), v(y,z)
'''
####################################
print ' ' 
print '#############################'
print 'CREATING INITIAL CONDITONS'
print '#############################'
print ' ' 
b_IC = np.zeros([Ly,N])
b_IC, hbls_IC = TTTW_func.b_hbl_IC(grids, b0, N20, N2b, B, lh, lam_inv, h0, dh, prof_IC, dom_IC)
Kv_IC         = TTTW_func.Kv_IC_hbl(hbls_IC, h0, dh, K_bak, K0, sig0, z_u_r, dom_IC, K_choice)
Kt_IC  = np.copy(Kv_IC)
hbbl_IC = np.zeros(len(hbls_IC))
ghat_IC = np.zeros(len(hbls_IC))
###############################################
# CHECK FOR PRELOADED FILES FOR VERTICAL MIXING
###############################################
if K_choice == 'IDEAL_LOAD':
   path_load_K = './K_saved/'
   print 'PRELOADING Kv, Kt INITIAL CONDITONS...'
   K_save_dict = TTTW_func.load_pickle_out(path_load_K + 'K_dict.p')
   Kv_IC[:,:] = K_save_dict['Kv_nt'][0,:,:]
   Kt_IC[:,:] = K_save_dict['Kt_nt'][0,:,:]


if IC_b == 'IDEAL_LOAD':
   path_load_b = './ICS/'
   print 'PRELOADING b INITIAL CONDITON'
   IC_temp = TTTW_func.load_pickle_out(path_load_b + 'IC_dict.p')
   b_IC[:,:] = IC_temp['b_IC'][:,:]


#################################################################################################

#                                   SOLVE FOR INITIAL TTW FLOWS
'''
SOLVE FOR INITIAL CONDITION FLOWS AS FOLLOWS:


1) OBTAIN A FIRST GUESS FOR LONGTINDUINAL FLOW (u(y,z)) by thermal wind integration
2) CALCULATE THE TOTAL PRESSURE WITH THE FIRST GUESS OF HTE LONGTINDUINAL FLOW AND 
   THE BUOYANCY PROFILE
3) CALCULATE FIRST GUESS TTW FLOWS (u,v) WITH THE FULL PRESSURE, SURFACE STRESSES AND NO BOTTOM STRESS
4) CALCULATE A BOTTOM STRESS FROM THE FIRST GUESS TTW FLOWS
5) ITERATE THE TTW DIAGNOSTIC WITH THE BOTTOM STRESSES UNTIL CONVERGENCE (WITH NEW BOTTOM STRESSES
   AND TOTAL PRESSURES CALCUALTED WITH EACH ITERATION)


'''
#################################################################################################

def calc_bstress(u,v):
    ''''
    FUNCTION TO CALCULATE BOTTOM STRESSES GIVEN A u,v field
    '''
    CD =  TTTW_func.calc_CD(Zob,grids)
    Tbx_t, Tby_t = TTTW_func.CD_stress(u,v,Zob,grids,rho0)
    # DIVIDE BY RHO0 FOR USE IN MOMENTUM EQN
    #IF BOTTOM STRESS IS BEING USED IN SIMULATION, USE IT
    if bottom_stress:   
       bustr = Tbx_t / rho0 
       bvstr = Tby_t / rho0
    else:
       bustr = np.zeros([Ly])
       bvstr = np.zeros([Ly+1])
    return bustr, bvstr
    #########################################


######################################################
# BEFORE SOLVING FOR INITIAL TTW FIELDS (STEADY STATE)
# NEED TO SOLVE BAROTROPIC PRESSURE EQUATION TO
# OBATIN FULL PRESSURE GRADIENT
######################################################
# CALCULATE BAROCLINC PRESSURE
phi_IC_b = TTTW_func.calc_phi_baroclin(b_IC,grids,rho0,g)
#SET INITIAL PRESCRIPTION OF ubar (zero or TW=thermal wind depth-average)
if ubar_choice == 'zero':
   ubar_in = np.zeros([Ly])
if ubar_choice == 'TW':
   u_TW = TTTW_func.make_u_IC_for_pressure(b_IC,grids,f_arr)
   ubar_in = np.mean(u_TW,axis=1)

##########################################################################
# SOLVE PRESUSRE POISSON EQUATION WITH BAROTROPIC ICs to get FULL PRESSURE
##########################################################################
svstr_IC = np.zeros([Ly+1])
svstr_IC[:] = svstr0_IC
bvstr_IC = np.zeros([Ly+1])
phi_total_IC = TTTW_func.calc_phi_total(ubar_in,phi_IC_b,svstr_IC,bvstr_IC,f_arr,grids,rho0)
phi_grad_IC = TTTW_func.horiz_grad(phi_total_IC,y_u_r)
##############################################################################################




#########################################
# GET INITIAL U,V FROM STEADY
# STATE TTW (WITH ZERO BOTTOM STRESSES)
###########################################
# set coriolis, wind, bottom stresses
sustr = np.zeros([Ly])
svstr = np.zeros([Ly+1])
svstr[:] = svstr0_IC
sustr[:] = sustr0_IC
bustr   = np.zeros([Ly])
bvstr   = np.zeros([Ly+1])
ut0,vt0_temp,ug,vg = TTTW_func.steady_state_TTW(phi_grad_IC,Kv_IC,sustr,svstr,bustr,bvstr,f_arr,grids,bot_stress=1,timing=False)
vt0 = TTTW_func.u2v(vt0_temp)

############################################


##############################################
# CREATE DICTIONARY OF VARIABLES TO SAVE
############################################
var_IC_dict = {}
keys_save = ['u_IC', 'v_IC', 'Kv_IC', 'Kt_IC', 'hbls_IC', 'hbbl_IC', 'phi_total_IC','ghat_IC']
key_shapes = [ [Ly,N], [Ly+1,N], [Ly,N+1], [Ly,N+1], [Ly], [Ly],    [Ly,N],[Ly,N+1]]
for ke in range(len(keys_save)):
    var_IC_dict[keys_save[ke]] = np.zeros(key_shapes[ke])


##############################################
#############################################

# SAVE AND EXIT HERE IF KPP NOT USED 

###########################################
############################################
if K_choice == 'IDEAL' or K_choice == 'IDEAL_LOAD': 
   #SAVE LAST VELOCITIES FROM ITERATION
   var_IC_dict['u_IC']         = ut0
   var_IC_dict['v_IC']         = vt0
   var_IC_dict['phi_total_IC'] = phi_total_IC
  
   # SET MIXING ICs
   var_IC_dict['Kv_IC'] = Kv_IC
   var_IC_dict['Kt_IC'] = Kt_IC
   #var_IC_dict['hbls_IC'] = np.zeros([Ly])
   #var_IC_dict['hbbl_IC'] = np.zeros([Ly])
   #var_IC_dict['ghat_IC'] = np.zeros([Ly,N+1])
  
   #SET BUOYANCY INITIAL CONDITION
   var_IC_dict['b_IC'] = b_IC
   
   # CHANGE TO IC DIRECTORY, SAVE, AND EXIT
   TTTW_func.change_dir_py('ICS')
   TTTW_func.save_to_pickle(var_IC_dict, 'IC_dict')
   os.chdir('../')
   sys.exit()
   ##########################################################################




#######################################################
# ITERATE KPP UNTIL CONVERGENCE OF INITIAL CONDITONS
#######################################################
################################################
# SOME FUNCTIONS FOR ITERATION PROCEDURE
##############################################
def fill_iter_keys(keys_fill,dict_in, new_arrs,n_iter,relaxed=False):
    '''
    FUNCTION TO FILL KEYS OF ITERATIONS DICTIONARY
    WITH UPDATED VALUES (CAN BE RELAXED ITERATIVE FORM)

    dict_in --> dictionary with all arrays from previous iterations
    new_arrs --> newly calculated fields that are to be added (or relaxed)
    '''
    for ke in range(len(new_arrs)):
        if keys_fill[ke] == 'Kv' or keys_fill[ke] == 'Kt' or keys_fill[ke] == 'hbls' or keys_fill[ke] =='hbbl':
           if relaxed:
              #UPDATE KEY WITH RELAXED ITERATION OF VARIABLE
              dict_in[keys_fill[ke]].append( ((1-alpha_r_iters_kpp) * dict_in[keys_fill[ke]][n_iter]) + alpha_r_iters_kpp *new_arrs[ke])
           else:
              dict_in[keys_fill[ke]].append(new_arrs[ke])

        else:
            dict_in[keys_fill[ke]].append(new_arrs[ke])

    return dict_in
    #####################################################


def test_convergence(dict_in, n_iter, j_point, k_point):
    '''
    RETURN BOOLEANS TESTING CONVERGENCE OF Kv, v
    '''
    keys = ['Kv']
    bools=[]
    diffs = []
    for ke in range(len(keys)):
        diffs.append( abs(dict_in[keys[ke]][n_iter][j_point, k_point]  - dict_in[keys[ke]][n_iter-1][j_point,k_point]))
        bools.append(diffs[ke]<=K_diff_thresh)
    return bools, diffs
    ###########################

def update_fields(dict_in, n_iter):
    '''
    UPDATE TTW AND KPP FIELDS FOR AN ITERATION
    '''

    u    = iters_dict['u'][n_iter]
    v    = iters_dict['v'][n_iter]
    Kv   = iters_dict['Kv'][n_iter]
    Kt   = iters_dict['Kt'][n_iter]
    hbls = iters_dict['hbls'][n_iter]
    hbbl = iters_dict['hbbl'][n_iter] 

    #####################
    #CALCULATE TTW FLOW
    #####################


    bustr, bvstr = calc_bstress(u,v)
    ut_new, vt_new, ug, vg = TTTW_func.steady_state_TTW(phi_grad_IC, iters_dict['Kv'][n_iter], sustr, svstr,bustr, bvstr, f_arr, grids, bot_stress=1,timing=False) 

    ###############################
    # CALCULATE NEW TOTAL PRESSURE
    #############################
    ubar_in = np.mean(ut_new,axis=1)
    phi_total_new = TTTW_func.calc_phi_total(ubar_in,phi_IC_b, svstr, bvstr, f_arr, grids, rho0) 
    phi_grad_IC_new = TTTW_func.horiz_grad(phi_total_new,y_u_r)

    ##################################
    # RUN KPP
    ###################################
    KPP_obj = KPP.TTTW_KPP(b_IC, ut_new, TTTW_func.u2v(vt_new), hbls, hbbl,Kv,Kt, srflx, sustr, svstr,f_arr, grids, False, 0) 
    KPP_obj.run_KPP()

    #FILL ITERATION DICTIONARY
    keys_to_fill = ['u', 'v', 'Kv', 'Kt', 'hbls', 'hbbl','ghat', 'phi_total']
    new_fields = [ut_new, TTTW_func.u2v(vt_new), KPP_obj.Kv_out, KPP_obj.Kt_out, KPP_obj.hbls, KPP_obj.hbbl, KPP_obj.ghat,phi_total_new]
    dict_update = fill_iter_keys(keys_to_fill,dict_in, new_fields, n_iter, relaxed=True) 

    del KPP_obj

    return dict_update
    ################################################

############################################################################################################


###########################
#RUN INITIAL KPP ITERATION
##########################
Q = np.zeros([Ly])
Q[:] = Q0_IC
srflx = Q  / (rho0*Cp)
var_IC_dict['hbls_IC'] = hbls_IC
KPP_obj = KPP.TTTW_KPP(b_IC, ut0, vt0, var_IC_dict['hbls_IC'], var_IC_dict['hbbl_IC'], Kv_IC, Kt_IC, srflx, sustr, svstr,f_arr, grids, False, 0) 
KPP_obj.run_KPP()


####################################################
# CREATE DICTIONARY OF VARIABLES FOR KPP ITERATIONS
###################################################
iters_dict = {}
new_keys = ['u', 'v', 'Kv', 'Kt', 'hbls', 'hbbl','ghat', 'phi_total']
for ke in range(len(new_keys)):
    iters_dict[new_keys[ke]] = []

###################################
#FILL KEYS WITH INITIAL ITERATION
###################################
keys_fill      = ['u', 'v', 'Kv', 'Kt', 'hbls', 'hbbl', 'ghat', 'phi_total']
init_iter_fill = [ut0, vt0, KPP_obj.Kv_out, KPP_obj.Kt_out, KPP_obj.hbls, KPP_obj.hbbl, KPP_obj.ghat, phi_total_IC]
old_iter       = [ut0, vt0, Kv_IC, Kt_IC, hbls_IC, hbbl_IC, ghat_IC, phi_total_IC]

for ke in range(len(keys_fill)):
    iters_dict[keys_fill[ke]].append(old_iter[ke])


#######################################################
# PERFORM FIRST RELAXED ITERATION AND TEST CONVERGENCE
######################################################
n = len(iters_dict['u'])
print 'Iteration=', n-1
iters_dict = update_fields(iters_dict, n-1)

#######################################
# TEST CONVERGENCE
#######################################
results = test_convergence(iters_dict, n, Ly/2, -10)
bool1 = results[0][0]
diff1 = results[1][0]
 
n+=1
#while n<20:
while not (bool1):
      print 'Iteration #: ', n-1
      iters_dict = update_fields(iters_dict,n-1)      
      ####################
      #TEST CONVERGENCE
      ###################
      results = test_convergence(iters_dict, n, Ly/2, -20)
      bool1 = results[0][0]
      diff1 = results[1][0]
      print '|Kv[n+1] - Kv[n]| = ' + str(diff1)
      n+=1
      ################################################################



##################################################
# WRITE OUPUT TO DICTIONARY AND SAVE DICTIONARY
#################################################
#SAVE LAST VELOCITIES FROM ITERATION
var_IC_dict['u_IC']         = iters_dict['u'][-1]
var_IC_dict['v_IC']         = iters_dict['v'][-1]
var_IC_dict['phi_total_IC'] = iters_dict['phi_total'][-1]

# SET MIXING ICs
var_IC_dict['Kv_IC']   = iters_dict['Kv'][-1]
var_IC_dict['Kt_IC']   = iters_dict['Kt'][-1]
var_IC_dict['hbls_IC'] = iters_dict['hbls'][-1]
var_IC_dict['hbbl_IC'] = iters_dict['hbbl'][-1]
var_IC_dict['ghat_IC'] = iters_dict['ghat'][-1]

#SET BUOYANCY INITIAL CONDITION
var_IC_dict['b_IC'] = b_IC

# CHANGE TO IC DIRECTORY, SAVE, AND EXIT
TTTW_func.change_dir_py('ICS')
TTTW_func.save_to_pickle(var_IC_dict, 'IC_dict')
os.chdir('../')
sys.exit()


