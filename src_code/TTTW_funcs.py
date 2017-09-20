######################################

__title__          = "TTTW_funcs.py"
__author__         = "Daniel Dauhajre"
__date__           = "September 2017"
__email__          = "ddauhajre@atmos.ucla.edu"
__python_version__ = "2.7.9"

'''

LIBRARY OF FUNCTIONS FOR IDEALIZED T3W MODEL
INCLUDING SETTING UP OF THE GRID, CREATING / SOLVING
FOR INITIAL CONDITIONS, PACKAGING DATA INTO
DICTIONARIES TO SEND TO TIME-STEPPING MODULE

'''
######################################

##################################
#	 IMPORT LIBRARIES
###################################
import os
import sys
import numpy as np
import scipy as sp
from pylab import *
import matplotlib.pyplot as plt
from scipy import integrate
from copy import copy
import time as tm
from scipy.sparse import *
from scipy.sparse.linalg import spsolve
from scipy import special
from netCDF4 import Dataset
import pickle as pickle
from decimal import Decimal as Decimal
import multiprocessing as MP
#################################


############################################################
#             SAVE/LOAD DICTIONARY WITH PICKLE
############################################################
def change_dir_py(dir_name):
     if not os.path.exists(dir_name):
        print 'Creating directory: ' + dir_name
        os.makedirs(dir_name)
     print 'Moving into directory: ' + dir_name
     os.chdir(dir_name)
     ##########################################

def save_to_pickle(var_dict,out_name):
    pickle.dump(var_dict,open(out_name +".p", "wb"))
    print 'Saved output as: ' + out_name + '.p'
    ##########################################

def load_pickle_out(path_full):
    """
    LOAD PICKLED DICTIONARY
    path_full --> full path  including file name and  ".p" suffix
    """
    print ''
    print 'LOADING: ' + path_full
    print ''
    return pickle.load(open(path_full, "rb"))
    ##########################################

def add_keys_var_dict(keys_add,keys_type,var_dict={}):
    '''
    var_dict --> dictionary to add keys to (default is empty {} or
                 but can be an actual pre-existing dictionary)

    keys_add ---> list of string of keys to set up dictionary with

    keys_typ e--> list of data types corresponding to each key (i.e., empty
                       list or string or numpy array
    '''
    for ke in range(len(keys_add)):
        var_dict[keys_add[ke]] = keys_type[ke]

    return var_dict
    ###########################################################







###################################
# make_grid
'''
Creates 2D (y,z)grid from dimensions given in METERS
Ly --> length of horizontal axis (at rho-points)
Lz --> length of vertical axis
dy --> horizontal spacing
dz --> vertical spacing

All distances in meters

Horizontally and vertically staggered grid

'''
###################################


def make_grid(Ly,Lz,dy, dz):


        print ' '
	print 'Creating (y,z) grid with following dimensions:'
	print '#######################'
	print 'Ly = ' + str(Ly) +  'm'
	print 'Lz = ' + str(Lz) +  'm'
	print 'dy = ' + str(dy) +  'm'
        print 'dz = ' + str(dz) +  'm'
        print '#######################'
	
	
	
	# CREATE U,V,RHO,W ARRAYS
	y_u = np.arange(-Ly/2,Ly/2+dy,dy)
	y_v = np.arange((-Ly/2)-(dy/2.), (Ly/2) + (dy/2.)+dy ,dy)
	z_r = np.arange(-Lz+(dz/2.),dz/2.,dz)
	z_w = np.arange(-Lz,0+dz,dz)

        # CREATE MESHGRID FOR ALL POINTS
	z_u_r,y_u_r  = np.meshgrid(z_r,y_u)
	z_v_r, y_v_r = np.meshgrid(z_r,y_v)
	z_u_w, y_u_w = np.meshgrid(z_w,y_u)
	z_v_w,y_v_w  = np.meshgrid(z_w,y_v)
#
        grid_dict = {}
	grid_dict['y_u_r'] = y_u_r
	grid_dict['z_u_r'] = z_u_r
        grid_dict['y_v_r'] = y_v_r
	grid_dict['z_v_r'] = z_v_r
	grid_dict['y_u_w'] = y_u_w
	grid_dict['z_u_w'] = z_u_w
        grid_dict['y_v_w'] = y_v_w
	grid_dict['z_v_w'] = z_v_w
        
	Ly_gridpts = y_u_r.shape[0]
	N_gridpts  = y_u_r.shape[1]
        
	print ' '
        print 'Grid created...'
        print ' '
	print '######################################'
	print 'Number of horizontal u-points: ' + str(Ly_gridpts)	
	print 'Number of horizontal v-points: ' + str(Ly_gridpts+1)
        print 'Number of vertical rho-levels: ' + str(N_gridpts)
	print 'Number of vertical w-levels: '   + str(N_gridpts+1)
        print '######################################'
	
	return grid_dict, Ly_gridpts, N_gridpts
        ########################################################################







            #########################################################################################

            #                   CREATE INITIAL BUOYANCY FIELD PROFILES

            ##########################################################################################

def h_fil(y,h0,dh,lh):
    '''
    CREATE A 
    BOUNDARY LAYER PROFILE OF A FILAMENT
    '''
    Ly = len(y)
    hbl = np.zeros([Ly])
    for j in range(Ly):
        hbl[j] = h0 + (dh  * np.exp( -(float(y[j])/lh)**2))
    return hbl

def h_front(y,h0,dh,lh):
    '''
    CREATE A 
    BOUNDARY LAYER PROFILE OF A FRONT
    '''
    Ly = len(y)
    hbl = np.zeros([Ly])
    for j in range(Ly):
        hbl[j] = h0 - ((dh/2.) * special.erf(float(y[j])/lh))
    return hbl


def b_hbl_IC(grids, b0, N20, N2b, B, lh, lam_inv, h0, dh, prof, dom):
    '''
    CREATE INITIAL BUOYANCY AND BOUNDARY LAYER PROFILE
    BASED ON MCWILLIAMS SCFT TYPE IDEAL FRONT / FILAMENT


    grid_dict ---> dictionary for grid
    b0       ---> background buoyancy 
    N20      ----> interior stratification
    N2b     ---> background stratification
    B       ---> fractionally reduced stratification in SBL (0<B<1)
    lh      ---> width of the front or filament (meters)
    lam_inv  ---> vertical scale transition between surface and interior layers
    h0      ---> boundary layer depth
    dh     ----> change in surface boundary layer depth
    prof   ---> 'FRONT' or 'FIL'
    dom    ----> shallow or deep water ('SW' or 'DW')

                SW ---> sets a uniformly mixed profile based on surface b(y,z=0) value of stratified profile
                DW ---> surface boundary layer that transitions into stratified interior
    '''
    y_u_r = grids['y_u_r']
    z_u_r = grids['z_u_r']
    H     = abs(grids['z_u_w'][0,0])
    [Ly,N] = grids['y_u_r'].shape
    b = np.zeros([Ly,N])
  
    ##################################### 
    #CREATE BOUNDARY LAYER DEPTH PROFILE 
    #####################################
    if prof == 'FIL':
       hbl = h_fil(y_u_r[:,0], h0,dh,lh)
    if prof == 'FRONT':
       hbl = h_front(y_u_r[:,0], h0,dh,lh)

    ##########################
    #CREATE BUOYANCY PROFILE
    ############################
    for j in range(Ly):
        for k in range(N):
            zz = (1./lam_inv) * (z_u_r[j,k] + hbl[j])
            b[j,k] = b0 + N2b * (z_u_r[j,k] + H)\
                     + (N20/2.) * ( ((1 + B)*z_u_r[j,k]) \
                              - (1-B) * (hbl[j] + lam_inv*np.log(np.cosh(zz))))



    ######################################
    # CREATE VERTICALLY UNIFORM PROFILE
    # IF SHALLOW-WATER (SW) DOMAIN CHOICE
    #######################################
    if dom == 'SW':
       b_out = np.zeros([Ly,N])
       hbl = np.zeros([Ly])
       hbl[:] = -z_u_r[:,0]
       for k in range(N):
           b_out[:,k] = b[:,-1]
       return b_out, hbl

    return b, hbl
    ###############################################


def Kv_IC_hbl(hbl, h0, dh, K_bak, Km0, sig0, z_u_r,dom,K_choice):
    '''
    CREATE VERTICAL MIXING PROFILE BASED ON SURFACE BOUNDARY
    LAYER PROFILE
    hbl ---> surface boundary layer depth
    h0 --> surface boundary layer baseline depth
    dh ---> surface boundary layer depth variation (horizontal)
    K_bak --> background diffusivity (mixing)
    Km0 ---> maximum diffusivity (mixing)
    sig0 ---> constant 
    z_u_r---> rho-levels, u-point vertical grid
    dom    ----> shallow or deep water ('SW' or 'DW')

                SW ---> sets a uniformly mixed profile based on surface b(y,z=0) value of stratified profile
                DW ---> surface boundary layer that transitions into stratified interior
 
    '''
    [Ly,N] = z_u_r.shape
    Km0 = (Km0 - K_bak) / (1 + dh/h0)
    Km = np.zeros([Ly])
    for j in range(Ly):
        Km[j] = Km0 * hbl[j] / h0
        if dom == 'SW':
           Km[j] = Km0

    ff = 4./27 * (1+sig0)**2
    Kv = np.zeros([Ly,N+1])
    Kv[:,:] = K_bak
    for j in range(Ly):
        for k in range(1,N):
            sig = -z_u_r[j,k] / hbl[j]
            if sig<=1:
               Kv[j,k] = Kv[j,k] + Km[j] * (sig + sig0) * (1-sig)**2/ff
  
    if K_choice == 'IDEAL':
       Kv[:,:] = Km0
    return Kv 
    ############################################################


            #########################################################################################

            #                   CREATE INITIAL GUESSES FOR LONGITUDINAL FLOW

            ##########################################################################################


####################################
'''
CREATE INTIIAL GUESS OF ALONG-FILAMENT
FLOW PROFILE FROM BUOYANCY PROFILE
TO BE USED IN CALCULATION OF INITIAL
TOTAL PRESSURE

du/dz = -db/dy / f
integrate du/dz from bottom up
to get a surface-intense profile
'''
####################################
def make_u_IC_for_pressure(b_IC,grid_dict,f_arr):
    y_u_r = grid_dict['y_u_r']
    z_u_w = grid_dict['z_u_w']
    [Ly,N] = y_u_r.shape

    # CALCULATE VERTICAL SHEAR VIA THERMAL WIND
    db_dy = horiz_grad(b_IC,y_u_r)
    uz  = np.zeros([Ly,N])
    for k in range(N):  
        uz[:,k] = -db_dy[:,k] / f_arr

    #INTEGRATE uz up from bottom to get u_IC
    Hz = z_u_w[:,1:] - z_u_w[:,:-1]
    u_IC = np.zeros([Ly,N])
    u_IC[:,0] = 0.5 * Hz[:,0] * uz[:,0]
    for k in range(N-1):
        u_IC[:,k+1] = u_IC[:,k] + (0.5 * ((Hz[:,k] * uz[:,k]) + (Hz[:,k+1]*uz[:,k+1])))

    return u_IC
    ##################################################################################





####################################################################
# CALCULATE BAROCLININC PRESSURE(really geopotential)  (HYDROSTATIC)
'''
CALCULATE BAROCLINC PRESSURE (NORMALIZED BY RHO0)
INPUTS:
b --> 2D[Ly,N]: buoyancy field
grid_dict ---> dictionary of gridded meshes
rho0 --> reference density (double)
g --> graviational constant (double)

OUTPUTS:
phi --> 2D[Ly,N]: hydrostatic pressure (normalized by rho0)
'''
###################################################################
def calc_phi_baroclin(b,grid_dict,rho0,g):
    z_u_r = grid_dict['z_u_r']
    z_u_w = grid_dict['z_u_w']
    Ly,N = b.shape
    phi = np.zeros([Ly,N])
    Hz = z_u_w[:,1:] - z_u_w[:,:-1] 
    # INTEGRATE FROM TOP DOWN USING TRAPEZOIDAL RULE
    b_minus_g = -(b)
    phi[:,-1] = 0.5 * Hz[:,-1] * b_minus_g[:,-1]
    #phi[:,-1] = 0.
    for k in range(N-1,0,-1):
        phi[:,k-1] = phi[:,k] + ( 0.5 * ((Hz[:,k] * b_minus_g[:,k]) + (Hz[:,k-1] * b_minus_g[:,k-1])))
    return phi
    ################################################################


###############################################
# calc_phi_total()
'''
CALCULATE TRUE BAROTROPIC PRESSURE
BY SOLVING THE PRESSURE POISSON EQUATION
FOR THE TTTW SYSTEM

INPUTS:
ubar --> 2D [Ly]: x-direction barotropic velocity (either actual velocity at each time step or prescribed IC velocity)
phi_b --> 2D [Ly,N]: baroclinic pressure calculated hydrostatically (from calc_phi_baroclin)
svstr --> 1D: [Ly+1]: surface wind stress in y-direction ( m^2/s^2)
bvstr --> 1D: [Ly+1]: bottom stress in y-direction (m^2/s^2)
f --> 1D: [Ly]: coriolis paramter 
grid_dict --> dictionary of gridded meshes
rho0 ---> double: reference density

OUTPUTS:
phi_total --> 2D [Ly,N]: full pressure with true barotropic pressure added in as correction

'''



################################################

def calc_phi_total(ubar,phi_b, svstr, bvstr,f,grid_dict,rho0,BC_choice = 'doub_dirch'):
    z_u_w = grid_dict['z_u_w']
    y_u_r = grid_dict['y_u_r']
    [Ly,N] = y_u_r.shape
    H = -z_u_w[0,0]
    y_u_H = np.mean(y_u_r, axis=1)
    
    svstr_upts = v2u(svstr)
    bvstr_upts = v2u(bvstr)
    ############################
    # CONSTRUCT RHS FUNCTION
    ############################
    phi_b_bar = np.mean(phi_b,axis=1,dtype=np.float64)
    #ubar      = np.mean(u,axis=1)
    dy_H      = np.gradient(y_u_H)
 
    G2 = f * ( (np.gradient(ubar) / dy_H))
    G3 = (1./(H)) * (np.gradient(svstr_upts - bvstr_upts) / dy_H)

    G = -G2 + G3

    ##################################
    # CONSTRUCT COEFFICIENT MATRIX
    ##################################    
    ndim = Ly
    A = lil_matrix((ndim,ndim))
    R = np.zeros(ndim)

    # SIDE BOUNDARY
    idx = 0
    if BC_choice == 'neum_dirch':
       A[idx,idx] = -1
       A[idx,idx+1] = 1.
    if BC_choice == 'doub_dirch':
       A[idx,idx] = -2.
       A[idx,idx+1] = 1.
    
    # INTERIOR
    for j in range(1,Ly-1):
        idx = j
        A[idx,idx] = -2.
        A[idx,idx-1] = 1.
        A[idx,idx+1] = 1.
    
    # FAR SIDE BOUNDARY
    idx = Ly-1
    A[idx,idx] = -2.
    A[idx,idx-1] = 1.

    ############################
    # CONSTRUCT RHS MATRIX
    #############################
    for j in range(Ly):
        R[j] = G[j] * dy_H[j]**2


    # BOUNDARY CONDITIONS
    BC_neum  = -f[0] * ubar[0] * dy_H[0]
    phi0 = phi_b_bar[0]
    phi1 = phi_b_bar[-1]
    
    if BC_choice == 'doub_dirch':
       R[0] = (G[0] * dy_H[0]**2) - phi0
    if BC_choice == 'neum_dirch':
       R[0] = (G[0] * dy_H[0]**2 * 0.5) + BC_neum
   
    R[-1] = (G[-1] * dy_H[-1]**2) - phi1


    ######################
    # SOLVE SYSTEM
    ######################
    #CONVERT TO DECIMALS
    #R_dec = [Decimal(R[j]) for j in range(len(R))]
    A = A.tocsr()
    phi_truebar = spsolve(A,R)
   

    #######################################################
    # CORRECT BAROCLINIC PRESSURE AND RETURN FULL PRESSURE
    #########################################################
    phi_total = np.zeros([Ly,N],dtype=float64)
    phi_star = np.zeros([Ly,N],dtype=float64)
    for k in range(N):
        phi_star[:,k] = phi_b[:,k] - phi_b_bar 
        phi_total[:,k] = phi_star[:,k] + phi_truebar

    return phi_total
    ################################################################


            #########################################################################################

            #                   FUNCTIONS FOR VERTICAL VELOCITY AND ENFORCING CONTINUITY

            ##########################################################################################

##################################
# CALCULATE VERTICAL VELOCITY
# FROM CONTINUITY
'''
INPUTS:
v---> 2D [Ly+1,N]: velocity in y-direction AT V-POINTS
grid_dict ---> dictionary of grid meshes
'''
#################################

def calc_w(v,grid_dict):
    '''
    CALCULATE VERTICAL VELOCITY

    '''
    [Ly,N] = grid_dict['y_u_r'].shape
    y_v_w = grid_dict['y_v_w']
    z_v_w = grid_dict['z_v_w']
    z_u_w = grid_dict['z_u_w']    
    y_v_r = grid_dict['y_v_r']

    # FINITE VOLUME CONTINIUTY (ASSUME dV/dt = 0, volume of grid boxes does not change)    
    dy_u_w = y_v_w[1:,:] - y_v_w[:-1,:]
    dz     = z_u_w[:,1:] - z_u_w[:,:-1]
    w = np.zeros([Ly,N+1])
    #CALCULATE v-difference
    #dv = MP_dv(v) 
    dv = v[1:,:] - v[:-1,:]
    for k in range(1,N):
        kprime_end = k
        dv_sum = np.zeros([Ly])
        #dv_sum = (v[j+1,0] - v[j,0]) * dz[j,0]
        for kprime in range(kprime_end):
            dv_sum = dv_sum + ( (dv[:,kprime] )* dz[:,kprime])
        w[:,k] = (-1./dy_u_w[:,k]) * dv_sum
   
    return w
    ###########################################################


            #########################################################################################

            #                   BOTTOM STRESS CALCULATION FUNCTIONS

            ##########################################################################################


def calc_bstress(u,v,Zob, grid_dict, rho0, bot_stress):
    '''
    CALCULATE BOTTOM STRESSES GIVEN CHOICE OF BOTTOM 
    STRESS AND PUT IN CORRECT UNITS [m^2/s^2]
    '''
    if bot_stress:
       Tbx, Tby = CD_stress(u,v,Zob, grid_dict, rho0)
       return Tbx / rho0, Tby / rho0
    else:
       return np.zeros(u.shape[0]), np.zeros(v.shape[0])
    #########################################################



############################
# CD_stress()
'''
Calculate bottom stress
using bulk drag coefficeint
formulation

Separate function to calculate CD
so that it can be called in KPP
'''
###########################
def calc_CD(Zob,grids):
    # CALCULATE DRAG COEFFICIENT
    # FROM ROUGHNESS HEIGHT
    # AND GRID SPACING
    vonKar = 0.41
    z_u_r = grids['z_u_r'] 
    z_u_w = grids['z_u_w']

    zref = z_u_r[0,0] - z_u_w[0,0]
    CD = vonKar **2 / (np.log(zref/Zob))**2
    #print 'CD = ', str(CD)
    return CD


def CD_stress(u,v,Zob,grids,rho0):
    CD = calc_CD(Zob,grids)
  
  
    uref =  u[:,0]
    vref =  v[:,0]
    umag = np.sqrt(uref**2 + v2u(vref)**2)
    Tbx = (CD * umag * uref) * rho0 
    Tby = u2v((CD * umag * v2u(vref))) * rho0 

    return Tbx, Tby


def get_r_D(u,v,Zob,grid_dict):
    '''
    FOR COMPATABILITY WITH KPP in TTTW_kpp.py
    '''
    z_u_r = grid_dict['z_u_r']
    z_u_w = grid_dict['z_u_w']

    Hz = z_u_w[:,1:] - z_u_w[:,:-1]
    [Ly,N] = z_u_r.shape

    r_D = np.zeros([Ly])
    vonKar = 0.41

    v_upts = v2u(v)


    for j in range(Ly):
       cff = np.sqrt( u[j,0]**2 + ( 0.5 * (v[j,0] + v[j+1,0]))**2)

       r_D[j] = cff * (vonKar / (np.log(1+0.5 * Hz[j,0]/Zob)))**2
    return r_D
    ###########################################



#########################################
# lmd_swr_frac()
'''
Compute fraction of shortwave solar flux
pennetrating to the specified depth
due to exponential decay in Jerlov
water type using Paulson and
Simpson (1977) two-wavelength-band solar
absorption model 

TAKEN FROM lmd_swr_frac.F in ROMS

'''

#########################################
def lmd_swr_frac(grid_dict):
    z_u_w= grid_dict['z_u_w']
    z_u_r = grid_dict['z_u_r']

    [Ly,N] = z_u_r.shape


    # SET RECIPROCAL OF ABSORTPION COEFFICIENTS
    mu1 = np.zeros([5])
    mu1[0] = 0.35
    mu1[1] = 0.6
    mu1[2] = 1.
    mu1[3] = 1.5
    mu1[4] = 1.4

    mu2 = np.zeros([5])
    mu2[0] = 23.
    mu2[1] = 20.
    mu2[2] = 17.
    mu2[3] = 14.
    mu2[4] = 7.9

 
    #FRACTION OF TOTAL RADIANCE FOR
    # WAVELENGTH BAND 1 AS A FUNCTION
    # OF JERLOV WATER TYPE (fraction for band 2
    # is always r2=1-r1)
    r1 = np.zeros([5])
    r1[0] = 0.58
    r1[1] = 0.62
    r1[2] = 0.67
    r1[3] = 0.77
    r1[4] = 0.78

    # SET JERLOV WATER TYPE (TAKEN FORM L4PV PARAMTERS)
    Jwt = 1    
    Jwt_py = Jwt - 1  #acts as index for above paramaters

    attn1 = -1./mu1[Jwt_py]
    attn2 = -1./mu2[Jwt_py]


    # FRACTIOSN FOR EACH SPECTRAL BAND
    swdk1 = np.zeros([Ly])
    swdk2 = np.zeros([Ly])
    swr_frac = np.zeros([Ly,N+1]) #swr_frac at w-levels

    Hz = z_u_w[:,1:] - z_u_w[:,:-1]

    for j in range(Ly):
        # SET FRACTIONS FOR EACH SPECTRAL BAND
        # THEN ATTENUATE THEM SEPARATELY THROUGHOUT
        # THE WATER COLUMN
        swdk1[j] = r1[Jwt_py]
        swdk2[j] = 1. - swdk1[j]
        
        swr_frac[j,-1] = 1.
     
        # LOOP MATCHES THAT IN lmd_swr_frac.F
        # DIFFERENT THAN USUAL TRANSLATION
        # B/C INDEXING AT k_w-1
        for k in range((N+1)-1,0,-1):
            k_r = k - 1
            k_w = k
            xi1 = attn1 * Hz[j,k_r]
            if xi1 > -20.:
               swdk1[j] = swdk1[j] * np.exp(xi1)
            else:
                swdk1[j] = 0.

            xi2 = attn2 * Hz[j,k_r]
            if xi2 > -20:
               swdk2[j] = swdk2[j] * np.exp(xi2)
            else:
               swdk2[j] = 0.

            swr_frac[j,k_w-1] = swdk1[j] + swdk2[j]
    return swr_frac
    ###############################################################


###########################################
# TTW SOLVER FOR IC
# FUNCTION TO GET INITIAL PROFILES OF U,V
# BY DOING STEADY-STATE TTW CALCULATION

# def steady_state_TTW(py,AKv,sustr,svstr,Tbx,Tby,z_r,z_w,timing=False)

'''
SOLVES STEADY STATE TTW FOR IDEALIZED (Y,Z) SYSTEM

INPUTS:
py --> 2D [Ly,N]: pressure gradient in y-direction
AKv--> 2D [Ly,N+1]: vertical mixing coefficient
sustr ---> 1D [Ly]: x-component surface wind stress
svstr --> 1D [Ly+1]: y-component surface wind stress
bustr  ---> 1D[Ly]: x-component of bottom stress
bvstr ---> 1D[Ly+1]: y-component of bottom stres
grid_dict -->dictionary containing gridded meshes
f   -----> coriolis paramter, 1D [Ly]
bot_stress ---> use bottom b.c with bottom stress (1=yes, 0=no)


OUTPUTS:
ut,vt --> TTW velocities (at vertical rho-levels and horizontal u-points)
ug,vg --> geostrophic velocities (at vertical rho-levels and horizontal u-points)

THE TTW SYSTEM IS SOLVED USING A TRIDIAGONAL MATRIX FOR A COUPLED (u,v) SYSTEM

Solves Ax = B where x = (u)
                        (v)

VERTICAL GRID STAGGERED (w and rho-levels)
with 
z_w[-1,-1] ---> surface
z_w[0,0,] ---> bottom

AKv formally indexed at 1/2...N+1/2 
AKv[:,0] ---> AKv_1/2 and AKv[:,-1] ---> AKv_(N+1/2)
Where N is number of vertical rho-levels

'''
##########################################
def steady_state_TTW(py,AKv,sustr,svstr,bustr,bvstr,f, grid_dict, bot_stress, timing=False):
    if timing: tstart = tm.time()
    z_r = grid_dict['z_u_r']
    z_w = grid_dict['z_u_w']

     
    # TRANSLATE Y-COMPONENT STRESSES TO U-POINTS
    svstr_upts = v2u(svstr)
    bvstr_upts   = v2u(bvstr)
    
    new = np.zeros(py.shape)
    
    #GET SYSTEM DIMENSIONS
    [ny,nz] = py.shape

    ###############################
    #VERTICAL DIFFERENCES
    '''
    dz vector at w-levels (difference btwn rho-levels)
    dz should be positive everywhere
    dz shape is [ny,nz-1]
    '''
    ###############################
    dz = z_r[:,1:] - z_r[:,:-1]
    Hz = z_w[:,1:] - z_w[:,:-1]
    ##########################
    # DECLARE SOLUTION ARRAYS
    #########################
    ut,vt = copy(new), copy(new)


    #############################
    #CREATE [A] Matrix
    #############################
    
    # COUPLED SYSTEM: has shape [2*nz,2*nz]
    ndim = 2 *nz
    kv = 2 #2 variable coupled system

    # SOLVE IN EACH VERTICAL COLUMN, SO LOOP IN HORIZONTAL AND SOLVE AT 
    # EACH HORIZONTAL LOCATION
    for j in range(ny):
	#if j%10==0: print 'Solving steady state TTW momentum equation: ', round(100.*j/(ny-1)), ' %'

	A = lil_matrix((ndim, ndim))
	R = np.zeros(ndim)
        
	##########################
        # 	BOTTOM
	'''
	Bottom coefficeint (C_3/2) 
	3/2 in discretization pertains to AKv[:,0+1]
	and dz[:,3/2] would be dz[:,0] for this formulation
	b/c dz is at rho-levels
	'''
	#########################
        idx = 0
	C5 = AKv[j,1]/dz[j,0]

	#FILL FIRST ROW
	A[idx,idx+1]  = -f[j] * Hz[j,0]
	A[idx,idx+kv] = -C5
	A[idx,idx]    = C5
        
        #FILL SECOND ROW
	A[idx+1,idx]      = f[j] *Hz[j,0]
	A[idx+1,idx+1+kv] = -C5
	A[idx+1,idx+1]    = C5
	


        #####################
	#     INTERIOR
	######################
        for k in range(1,nz-1):	    
	    ############################################    
	    # mapping AKv k-index from rho-level k-index
	    #############################################
	    # in this loop:
	    # if formal indexing is: AKv[k-1/2] --> translates to AKv[k]
	    # if formal indexing is: AKv[k+1/2] --> translates to AKv[k+1]
	    # where k is the rho-level indexer (loop indexer)

	    # dz is separately mapped from Akv since it is differences between  rho-levels
	    # formal indexing: dz[k-1/2] --> dz[k-1]
	    # formal indexing: dz[k+1/2] --> dz[k]
            
	 
	    C3 = AKv[j,k] / dz[j,k-1]
	    C4 = AKv[j,k+1] / dz[j,k]
	    C2 = C3 + C4

            idx = 2*k
	    
	    #FILL FIRST ROW
	    A[idx,idx+kv] = -C4
	    A[idx,idx]    = C2
            A[idx,idx-kv] = -C3
	    A[idx,idx+1]  = -f[j] * Hz[j,k]

	    #FILL SECOND ROW
	    A[idx+1,idx+1+kv] = -C4
	    A[idx+1,idx+1]    = C2
            A[idx+1,idx+1-kv] = -C3
	    A[idx+1,idx]      = f[j] *Hz[j,k]



	######################
        #        TOP
	#######################
        idx = (2*nz) - kv
        #C1 = AKv[N-1/2]/dz[N-1/2] ---> AKv[N-1/2] = AKv[nz-1] where nz is # of rho-levels
	#AKV[N-1/2] should be AKv at level below free surface (so it is non-zero)
	C1 = AKv[j,nz-1]/dz[j,-1]
       
        # TESTING NO STRESS CONDITION W/ GHOST POINTS
        # dz here is actually not physical, but it is a ghost cell so assume equal spacing 
        # as level below
        C0 = AKv[j,nz] / dz[j,-1]        

        #FILL FIRST ROW
        A[idx,idx-kv] = -C1
        A[idx,idx]    = C1
	A[idx,idx+1]  = -f[j] *Hz[j,-1]

	#FILL SECOND ROW
	A[idx+1,idx+1-kv] = -C1
	A[idx+1,idx+1]    = C1
	A[idx+1,idx]      = f[j]*Hz[j,-1]




        ######################################################################################
        ######################################################################################

        
	################################
        # 	FILL RHS MATRIX
	###############################
        for k in range(nz):
	    idx = 2 *k
	    R[idx] = 0 #in idealized model, no x-gradients
	    R[idx+1] = -py[j,k] * Hz[j,k]

        ####################
	#BOUNDARY CONDITIONS
        ###################

	#SURFACE B.C.
	R[-2] =  sustr[j]
	R[-1] =  -py[j,-1]*Hz[j,-1] +svstr_upts[j]

	#BOTTOM B.C.
	if bot_stress == 1:
           R[0] =  -bustr[j]
	   R[1] = -py[j,0]*Hz[j,0]- bvstr_upts[j]
           
	   #TESTING SOMETHING
	   #R[0] = 0
	   #R[1] = 0
          
        
        if timing: print 'Matrix definition OK.....', tm.time()-tstart
	if timing: tstart = tm.time()
        
        ##################################################################
        # 		     SOLVE TRIDIAG SYSTEM FOR u,v
	##################################################################
        #return A,R
	A = A.tocsr()
        
	if timing: print 'Starting computation.....', tm.time() - tstart
	if timing: tstart = tm.time()

	X = spsolve(A,R)
	if timing: print'...computation OK.......', tm.time() - tstart
	if timing: tstart = tm.time()
        
	#if j == 28:
	#   return A,R,X
	###################################################################

	##########################
        # REORDER RESULTS IN [j,k]
	#########################
        for k in range(nz):
	    idx = 2 *k
	    ut[j,k] = X[idx];
	    vt[j,k] = X[idx+1];

	if timing: print 'Allocaiton OK.........', tm.time() - tstart
	if timing: tstart = tm.time()

    ############################################
    # 	SOLVE FOR GEOSTROPHIC VELOCITIES
    ###########################################
    ug, vg = copy(new), copy(new)
    vg[:,:] = 0 # b/c px=0 in idealized model
    for j in range(ny):
        for k in range(nz):
	    ug[j,k] = -py[j,k] / f[j]
  
    return ut, vt, ug, vg
    ############################################################

###############################
# GET TIME VECTOR IN SECONDS
'''
dt_min --> time step in minutes
tend_days ---> end day of iteration (days)
'''
###############################
def get_tvec_sec(dt_min,tend_days):
    dt = dt_min * 60
    tend_sec = tend_days * 24 * 60 * 60.
    tvec_sec = np.arange(0,tend_sec,dt)
    return tvec_sec

###########################################
# write_output_nc_V2()
'''
Write output and all model paramters to netcdf
file

V2 --> asscoiated with dictionary data structures
       used in TTTW_timestepping_V2
'''
###########################################
def write_output_nc_V2(out_name,var_out_dict, grid_dict, consts_dict, tstep_dict,K_nt_dict,surf_flux_dict,choices_dict):
    
    ##################################
    # UNPACK DICTIONARIES
    #################################
    y_u_r = grid_dict['y_u_r']
    y_v_r = grid_dict['y_v_r']
    z_u_r = grid_dict['z_u_r']
    z_v_r = grid_dict['z_v_r']
    y_u_w = grid_dict['y_u_w']
    y_v_w = grid_dict['y_v_w']
    z_u_w = grid_dict['z_u_w']
    z_v_w = grid_dict['z_v_w']
    
    advect_bool    = choices_dict['advect_bool']
    bottom_stress     = choices_dict['bottom_stress']
    wind_tseries   = choices_dict['wind_tseries']
    Q_tseries      = choices_dict['Q_tseries']
    K_tseries      = choices_dict['K_tseries']


    f    = consts_dict['f']
    rho0 = consts_dict['rho0']  
    Dh    = consts_dict['Dh']
    alpha = consts_dict['alpha']
    Zob   = consts_dict['Zob']
 
    dt = tstep_dict['dt']
   
    K_choice = K_nt_dict['choice']

    sustr = surf_flux_dict['sustr']
    svstr = surf_flux_dict['svstr']
    Bfsfc = surf_flux_dict['Bfsfc']


    #DECLARE NETCDF FILE 
    out_file = Dataset(out_name + '.nc', 'w')

    [nt,ny,nz] = var_out_dict['u'].shape
    #CREATE DIMENSIONS  
    out_file.createDimension('time', nt)
    out_file.createDimension('ny', ny)
    out_file.createDimension('ny_v', ny+1)
    out_file.createDimension('nz_rho', nz)
    out_file.createDimension('nz_w', nz+1)



    ########################
    # SET GLOBAL ATTRIBUTES
    #########################
    out_file.title = out_name
    out_file.description = 'TTTW Idealized Simulation'
    out_file.Kchoice     = K_choice
    out_file.dt          = dt
    out_file.rho0        = rho0
    out_file.Zob         = Zob
    out_file.Dh          = Dh
    out_file.alpha       = alpha
    out_file.advect_bool    = str(advect_bool)
    out_file.bottom_stress  = str(bottom_stress)
    out_file.wind_tseries   = wind_tseries
    out_file.Q_tseries      = Q_tseries
    out_file.K_tseries      = K_tseries
    #####################################
    # DEFINE OUTPUT VARIABLES FOR NETCDF
    ####################################
    
    # WRITE GRIDS FOR PLOTTING IN POST-PROCESSING
    y_u_r_out = out_file.createVariable('y_u_r', dtype('float64').char, ('ny','nz_rho'))
    setattr(y_u_r_out, 'long_name', 'y_u_r grid')
    
    y_v_r_out = out_file.createVariable('y_v_r', dtype('float64').char, ('ny_v','nz_rho'))
    setattr(y_v_r_out, 'long_name', 'y_v_r grid')
    
    z_u_r_out = out_file.createVariable('z_u_r', dtype('float64').char, ('ny','nz_rho'))
    setattr(z_u_r_out, 'long_name', 'z_u_r grid')
    
    z_v_r_out = out_file.createVariable('z_v_r', dtype('float64').char, ('ny_v','nz_rho'))
    setattr(z_v_r_out, 'long_name', 'z_v_r grid')
    
    y_u_w_out = out_file.createVariable('y_u_w', dtype('float64').char, ('ny','nz_w'))
    setattr(y_u_w_out, 'long_name', 'y_u_w grid')
    
    y_v_w_out = out_file.createVariable('y_v_w', dtype('float64').char, ('ny_v','nz_w'))
    setattr(y_v_w_out, 'long_name', 'y_v_w grid')
    
    z_u_w_out = out_file.createVariable('z_u_w', dtype('float64').char, ('ny','nz_w'))
    setattr(z_u_w_out, 'long_name', 'z_u_w grid')
    
    z_v_w_out = out_file.createVariable('z_v_w', dtype('float64').char, ('ny_v','nz_w'))
    setattr(z_v_w_out, 'long_name', 'z_v_w grid')
    
    

    # DYNAMICAL FIELDS
    b_out = out_file.createVariable('b', dtype('float64').char, ('time','ny','nz_rho'))
    setattr(b_out, 'units', 'm/s^2')
    setattr(b_out, 'long_name', 'buoyancy')
 
    b_stagger_out = out_file.createVariable('b_stagger', dtype('float64').char, ('time','ny','nz_rho'))
    setattr(b_out, 'units', 'm/s^2')
    setattr(b_out, 'long_name', 'buoyancy staggered time')
    
    u_out = out_file.createVariable('u', dtype('float64').char, ('time', 'ny', 'nz_rho'))
    setattr(u_out, 'units', 'm/s')
    setattr(u_out, 'long_name', 'x-direction velocity')
    
    v_out = out_file.createVariable('v', dtype('float64').char, ('time', 'ny_v', 'nz_rho'))
    setattr(v_out, 'units', 'm/s')
    setattr(v_out, 'long_name', 'y-direction velocity')
    
    w_out = out_file.createVariable('w', dtype('float64').char, ('time', 'ny', 'nz_w'))
    setattr(w_out, 'units', 'm/s')
    setattr(w_out, 'long_name', 'z-direction velocity')
    
    Kv_out = out_file.createVariable('Kv', dtype('float64').char, ('time', 'ny', 'nz_w'))
    setattr(Kv_out, 'units', 'm^2/s')
    setattr(Kv_out, 'long_name', 'Vertical viscosity coefficient for momentum')

    Kt_out = out_file.createVariable('Kt', dtype('float64').char, ('time', 'ny', 'nz_w'))
    setattr(Kt_out, 'units', 'm^2/s')
    setattr(Kt_out, 'long_name', 'Vertical viscosity coefficient for buoyancy')

    ghat_out = out_file.createVariable('ghat', dtype('float64').char, ('time', 'ny', 'nz_w'))
    setattr(ghat_out, 'units', '1/s^2')
    setattr(ghat_out, 'long_name', 'KPP nonlocal transport term for buoyancy')


    hbls_out = out_file.createVariable('hbls',dtype('float64').char,('time','ny'))
    setattr(hbls_out, 'units', 'm')
    setattr(hbls_out, 'long_name', 'Surface boundary layer depth (positive)')

    hbbl_out = out_file.createVariable('hbbl',dtype('float64').char,('time','ny'))
    setattr(hbbl_out, 'units', 'm')
    setattr(hbbl_out, 'long_name', 'Bottom boundary layer depth (positive)')


    f_out = out_file.createVariable('f', dtype('float64').char, ('ny'))
    setattr(f_out, 'units', '1/s')
    setattr(f_out, 'long_name', 'Coriolis paramter')

    sustr_out = out_file.createVariable('sustr', dtype('float64').char, ('time','ny'))
    setattr(sustr_out, 'units', 'N/m^2')
    setattr(sustr_out, 'long_name', 'x-direction wind stress')

    svstr_out = out_file.createVariable('svstr', dtype('float64').char, ('time','ny_v'))
    setattr(svstr_out, 'units', 'N/m^2')
    setattr(svstr_out, 'long_name', 'y-direction wind stress')

    Bfsfc_out = out_file.createVariable('Bfsfc', dtype('float64').char, ('time','ny'))
    setattr(Bfsfc_out, 'units', 'W/m^2')#CHECK THIS!!!
    setattr(Bfsfc_out, 'long_name', 'Surface buoyancy flux')
    

    ###############################
    # FILL OUTPUT VARIABLES
    ##############################
    y_u_r_out[:,:] = y_u_r    
    y_v_r_out[:,:] = y_v_r    
    z_u_r_out[:,:] = z_u_r    
    z_v_r_out[:,:] = z_v_r    
    y_u_w_out[:,:] = y_u_w    
    y_v_w_out[:,:] = y_v_w    
    z_u_w_out[:,:] = z_u_w    
    z_v_w_out[:,:] = z_v_w   
   
    b_stagger_out[:,:,:] = var_out_dict['b_stagger'] 
    b_out[:,:,:]   = var_out_dict['b']
    u_out[:,:,:]   = var_out_dict['u']
    v_out[:,:,:]   = var_out_dict['v']
    w_out[:,:,:]   = var_out_dict['w']
    Kv_out[:,:,:]  = var_out_dict['Kv']
    Kt_out[:,:,:]  = var_out_dict['Kt']
    ghat_out[:,:,:] = var_out_dict['ghat']
    hbls_out[:,:]  = var_out_dict['hbls']
    hbbl_out[:,:]  = var_out_dict['hbbl']
    f_out[:]       = f
    sustr_out[:,:] = sustr
    svstr_out[:,:] = svstr
    Bfsfc_out[:,:] = Bfsfc    
        
    ############################
    # CLOSE FILE
    ###########################
    out_file.close()
    print 'Created and saved netcdf file: ' + out_name



################################
# horizontal point 
# conversion functions
# u2v --> u point to v-point

# v2u ---> v point to u point
##############################


###############
# v2u
###############

def v2u(var_v):
    if np.ndim(var_v)==2:
      var_u =  v2u_2D(var_v)
    elif np.ndim(var_v)==1:
       var_u = v2u_1D(var_v)
    return var_u

def v2u_1D(var_v):
    Lv = var_v.shape[0]
    Lu = Lv-1
    var_u = 0.5 * (var_v[0:Lu] + var_v[1:Lv])
    return var_u

def v2u_2D(var_v):
    [Lv,Nv] = var_v.shape
    Lu = Lv-1
    var_u = 0.5 * (var_v[0:Lu,:] + var_v[1:Lv,:])
    return var_u

###############
# u2v
###############

def u2v(var_u):
    if np.ndim(var_u)==2:
      var_v =  u2v_2D(var_u)
    elif np.ndim(var_u)==1:
       var_v = u2v_1D(var_u)
    return var_v

def u2v_1D(var_u):
    Lu = var_u.shape[0]
    Lv = Lu+1
    Lm = Lu -1
    var_v = np.zeros([Lv])
    var_v[1:Lu] = 0.5 * (var_u[0:Lm] + var_u[1:Lu])
    var_v[0] = var_u[0]
    #var_v[-1] = var_u[-1]
    #var_v[0]    = var_v[1]
    var_v[Lv-1] = var_v[Lu-1]
    return var_v

def u2v_2D(var_u):
    [Lu,N] = var_u.shape
    Lv = Lu+1
    Lm = Lu-1
    var_v = np.zeros([Lv,N])
    var_v[1:Lu,:] = 0.5 * (var_u[0:Lm,:] + var_u[1:Lu,:])
    var_v[0,:] = var_u[0,:]
    var_v[-1,:] = var_u[-1,:]
    #var_v[0,:]    = var_v[1,:]
    #var_v[Lv-1,:] = var_v[Lu-1,:]
    return var_v



###############
# w2rho
###############
def w2rho(var_w):
    if np.ndim(var_w) == 2:
       var_rho = w2rho_2D(var_w)
    elif np.ndim(var_w) == 1:
       var_rho = w2rho_1D(var_w)
    return var_rho

def w2rho_1D(var_w):
    N_w = var_w.shape[0]
    N   = N_w - 1
    var_rho = 0.5 * (var_w[0:N] + var_w[1:N_w])
    return var_rho

def w2rho_2D(var_w):
    [Ly,N_w] = var_w.shape
    N   = N_w - 1
    var_rho = 0.5 * (var_w[:,0:N] + var_w[:,1:N_w])
    return var_rho



###############
#rho2w 
###############

def rho2w(var_rho):
    if np.ndim(var_rho)==2:
      var_w =  rho2w_2D(var_rho)
    elif np.ndim(var_rho)==1:
       var_w = rho2w_1D(var_rho)
    return var_w

def rho2w_1D(var_rho):
    Nrho = var_rho.shape[0]
    Nw   = Nrho + 1
    Nm   = Nrho -1
    var_w = np.zeros([Nw])
    var_w[1:Nrho] = 0.5 * (var_rho[0:Nm] + var_rho[1:Nrho])
    var_w[0] = var_rho[0]
    var_w[-1] = var_rho[-1]
    #var_w[0] = var_w[1]
    #var_w[Nw-1] = var_w[Nrho-1]
    return var_w

def rho2w_2D(var_rho):
    Ly,Nrho = var_rho.shape
    Nw   = Nrho + 1
    Nm   = Nrho -1
    var_w = np.zeros([Ly,Nw])
    var_w[:,1:Nrho] = 0.5 * (var_rho[:,0:Nm] + var_rho[:,1:Nrho])
    var_w[:,0] = var_rho[:,0]
    var_w[:,-1] = var_rho[:,-1]
    #var_w[:,0] = var_w[:,1]
    #var_w[:,Nw-1] = var_w[:,Nrho-1]
    return var_w



###################################################
# SIMPLE HORIZONTAL GRADIENT FUNCTION
# FOR 2D FIELDS (only use for rho-pt) fields here
###################################################
def horiz_grad(field,y):
    dfield_dy = np.zeros(field.shape)
    if len(field.shape) == 2:
        for k in range(field.shape[1]):
            dfield_dy[:,k] = np.gradient(field[:,k]) / np.gradient(y[:,k])
    else:
        dfield_dy = np.gradient(field) / np.gradient(y)

    return dfield_dy
    #######################

