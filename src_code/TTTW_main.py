######################################

__title__          = "TTTW_main.py"
__author__         = "Daniel Dauhajre"
__date__           = "September 2017"
__email__          = "ddauhajre@atmos.ucla.edu"
__python_version__ = "2.7.9"

'''
Script that uses the functions in TTTW_funcs.py and TTTW_timestepping.py
to create and run a simulation based on initial conditions,
domain, diurnal forcing, etc.
'''
######################################

###################################
#       IMPORT LIBRARIES
###################################
import os
import sys
import numpy as np
import scipy as sp
from pylab import *
import TTTW_funcs as TTTW_func
import TTTW_timestepping as TTTW_tstep
import TTTW_kpp as KPP 
import pickle as pickle
##################################

#################################
# 	SET CONSTANTS
################################
code_path = './src_code/'
execfile(code_path+'params.py')

print '		################################################'
print '             		RUN ID: ' + run_ID  
print '		################################################'

print '                 #############################    '
print ' 		SIMULATION PARAMATERS/CHOICES    ' 
print '                 #############################    ' 
print ' 		advect_bool: ' + str(advect_bool)
print '                 bottom_stress: ' + str(bottom_stress)
print '                 K_choice: ' + K_choice 
print '                 horiz_diff: ' + str(horiz_diff)
print ' ' 
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



##################################################################


###############################################
# LOAD INITIAL CONDITONS FROM ITERATIVE SCHEME
#############################################
path_ICs = './' 'ICS/' 

print ' '
print 'Initial conditons loaded from: ' + path_ICs
print ' ' 


var_IC_dict = TTTW_func.load_pickle_out(path_ICs+'IC_dict.p')

################################
'''
LOAD INITIAL CONDTIONS
'''
##########################
Kv_IC = var_IC_dict['Kv_IC']
Kt_IC = var_IC_dict['Kt_IC']
u_IC = var_IC_dict['u_IC']
v_IC = var_IC_dict['v_IC']
b_IC = var_IC_dict['b_IC']
phi_total_IC = var_IC_dict['phi_total_IC']
hbls_IC = var_IC_dict['hbls_IC']
hbbl_IC = var_IC_dict['hbbl_IC']

## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#### ! HARDCODED NONLOCAL TERM FIX LATER!!!!! ####
ghat_IC   = np.zeros([Ly,N+1]) #HARCODED FOR NOW
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

##########################################
# FORM TIME VECTOR 
##########################################
tvec_sec = TTTW_func.get_tvec_sec(dt/60.,tend_days)
nt = len(tvec_sec)
#########################################################################
#           PACKAGE INPUT VARIABLES INTO DICTIONARIES TO
#           SEND INTO TIME-STEPPING MODULE
##########################################################################

#############################################
# DICTIONARY OF INITIAL CONDITIONS
###########################################
IC_keys        =['u', 'v', 'b', 'Kv', 'Kt', 'phi_total', 'hbls', 'hbbl', 'ghat']
IC_types       = [u_IC, v_IC, b_IC, Kv_IC, Kt_IC, phi_total_IC, hbls_IC, hbbl_IC, ghat_IC]
init_cond_dict = TTTW_func.add_keys_var_dict(IC_keys,IC_types,var_dict={})


########################################
# DICTIONARY OF CHOICES
######################################
choices_keys = ['advect_bool', 'bottom_stress', 'wind_tseries', 'Q_tseries', 'K_tseries', 'horiz_diff','tstep_scheme_main','MP_switch']
choices_types = [advect_bool, bottom_stress, wind_tseries, Q_tseries, K_tseries, horiz_diff,tstep_scheme_main,MP_switch]
choices_dict = TTTW_func.add_keys_var_dict(choices_keys,choices_types,var_dict={})


#################################################
# DICTIONARY OF CONSTANTS
##################################################
#CONSTANT ROTATION ACROSS Y-AXIS (CREATE ARRAY)
f_arr = np.zeros([Ly])
f_arr[:] = coriolis_param

const_keys  = ['rho0', 'f', 'g', 'alpha', 'Cp', 'Zob', 'Dh']
const_types = [rho0, f_arr, g,  alpha, Cp,Zob, Dh]
consts_dict = TTTW_func.add_keys_var_dict(const_keys,const_types,var_dict={})


###################################################
# TIME-STEPPING PARAMETERS
#################################################
tstep_keys = ['dt', 'tend_days', 'nt']
tstep_types = [dt, tend_days, nt]
tstep_dict = TTTW_func.add_keys_var_dict(tstep_keys, tstep_types, var_dict={})


#############################################
# CREATE FORCING TIME-SERIES AND STORE ALL
# SURFACE FORCING AND VERTICAL MIXING
# IN SEPARATE DICTIONARIES
############################################
#INITIALIZE VERTICAL MIXING TIME-SERIES
Kv_tseries = np.zeros([nt,Ly,N+1])
Kt_tseries = np.zeros([nt,Ly,N+1])
ghat_tseries = np.zeros([nt,Ly,N+1])
Kv_tseries[0,:,:] = Kv_IC
Kt_tseries[0,:,:] = Kt_IC

##################
#CREATE FORCINGS
#################
execfile(code_path + 'create_forcings.py')

##############################################
# FROM SOLAR HEAT FLUX CALCULAT BUOYANCY FLUX
###############################################
Bfsfc_nt = (Q_nt * alpha * g) / (rho0 * Cp)


####################################
# DICTIONARY OF VERTICAL MIXNG
#####################################
if K_choice == 'IDEAL_LOAD':
   path_load_K = './K_saved/'
   print 'PRELOADING Kv, Kt, ghat...'
   K_save_dict = TTTW_func.load_pickle_out(path_load_K + 'K_dict.p')
   Kv_tseries   = K_save_dict['Kv_nt'][:,:,:]
   Kt_tseries   = K_save_dict['Kt_nt'][:,:,:]
   ghat_tseries = K_save_dict['ghat_nt'][:,:,:]
 

K_keys = ['choice', 'Kv', 'Kt', 'ghat']
K_types = [K_choice, Kv_tseries, Kt_tseries, ghat_tseries]
K_nt_dict = TTTW_func.add_keys_var_dict(K_keys, K_types, var_dict={})

###############################
#DICTIONARY OF SURFACE FORCINGS
############################## 
surf_flux_keys = ['sustr', 'svstr', 'Bfsfc']
surf_flux_types = [sustr_nt, svstr_nt, Bfsfc_nt]
surf_flux_dict = TTTW_func.add_keys_var_dict(surf_flux_keys, surf_flux_types, var_dict={})


############################################
# FORM HORIZONTAL DIFFUSIVE CONSTANT ARRAY
############################################
Dh_arr = np.zeros([Ly+1])
Dh_arr[:] = Dh


##########################################################################################

####################################
# TIME STEPPING
###################################
var_out_dict = TTTW_tstep.tstep_main(grids, init_cond_dict, consts_dict, tstep_dict, K_nt_dict, surf_flux_dict, choices_dict, Dh_arr)




#########################
#WRITE OUTPUT TO NETCDF
###########################_
TTTW_func.change_dir_py('nc_files')
TTTW_func.write_output_nc_V2(run_ID,var_out_dict,grids,consts_dict,tstep_dict,K_nt_dict, surf_flux_dict,choices_dict)
os.chdir('../')





