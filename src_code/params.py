#######################################
run_ID = 'TEST'

#########################################
# SWITCHES (CHOICES)
'''
advect_bool   --> (True or False), if True, buoyancy advection turned on
bottom_stress --> (True or False), if True, bottom stress is calculated and used in momentum equation
K_choice      --> ('KPP' or 'IDEAL' or 'IDEAL_LOAD')
                     'KPP': KPP used 
                    'IDEAL': user must specify Kv(t,y,z) and Kt(t,y,z) in create_forcings.py
                    'IDEAL_LOAD': preloaded Kv(t,y,z) and Kt(t,y,z) (e.g., spatial averages of a KPP produced cycle) 

IC_b          --> ('' or 'IDEAL_LOAD)
                    '': buoyancy initial condition created as prescribed by parameters below
                    'IDEAL_LOAD': preloaded b(y,z) initial condition

horiz_diff    --> (True or False): horizontal diffusion turned on (True) or off (False)

preload_ideal --> (True or False) 

tstep_scheme_main --> 'AB3_SI' (use Adams-Bashforth 3rd order,semi-implicit time-stepping scheme for interior time points)
                      'LF_SI' (use leap-frog semi-implicit time-stepping for interior time points)

MP_switch    --> (True or False): if True, multiprocessing used to parallelize computations
'''
#########################################
advect_bool = True
bottom_stress = True
K_choice = 'KPP'
IC_b = ''
horiz_diff = True
tstep_scheme_main = 'AB3_SI'
MP_switch = False 


###############################
# FORCING TIME SERIES OPTIONS
'''
wind_tseries, Q_tseries, K_tseries
---> all strings, set by user
---> correspond to time-series of forcings created in create_forcings.py
---> can be edited by user to make whatever forcing prescription they want
'''
##############################
wind_tseries = 'V1_spindown_tozero'
Q_tseries= 'Kcomp_diurnal_linear_spin'
K_tseries = ''

##################################
# INITIAL CONDITONS PARAMETERS
'''
alpha_r_iters_kpp ---> relaxed iteration constant for
                      KPP iteration
0 < alpha_r < 1

K_diff_thresh --> threshold for iteration to converge
'''
##################################
alpha_r_iters_kpp = 0.01
K_diff_thresh = 1E-5


#########################################################
#	!!!!!!!!! PARAMATERS !!!!!!!!!!!!
#########################################################

###############################
# EVOLUTION EQUATION CONSTANTS
'''
rho0 --> reference density [kg m^-3]
g     ---> gravitational acceleration [ms^-2]
coriolis_param ---> rotation rate [s^-1]
alpha   ---> thermal expansion coeffcient
Cp     --- specific heat capacity [J/kg/degC]
Zob    ---> roughness height [m]
Dh     ---> horizontal diffusivity [ms^-2]
'''
################################
rho0 = 1027.5 
g = 9.81 
coriolis_param = 8E-5
alpha = 2E-4 
Cp = 3985
Zob = 1E-2
Dh = 36.

################################
# SPATIAL GRID CONSTANTS
'''
dy_m  --> horizontal grid cell spacing (meters)
Ly_m  --> length of horizontal axis (meters)
Lz_m  --> length of vertical axis (meters)
dz_m  --> vertical cell spacing (meters)
'''
################################
dy_m = 150 
Ly_m = 15000
Lz_m =  50
dz_m = 0.5

################################
# TEMPORAL GRID CONSTATNS
'''
dt        --> time-step (seconds)
tend_days --> length of simulation time (days) 
'''
################################
dt = 120 
tend_days = 9



###############################
# BUOYANCY, SBL, AND Kv PROFILE
# INITIAL CONDITION  PARAMETERS
'''
prof_IC --> 'front' or 'fil' (front or filament
dom_IC  --> 'DW' or 'SW' (deepwater or shallow-water)
             if 'DW', initial conditon will have stratification
             if 'SW', initial conditon will be fully mixed in vertical (i.e., no stratification)

FOLLOWIGN PARAMETERS ARE DETAILED IN:
McWilliams (2017): Submesoscale surface fronts and filaments: secondary circulation,
                buoyancy flux, and frontogenesis. Journal of Fluid Mechanics, (823), 391-432

lh     --> sets width of front of filament gradient (meters)
h0     --> baseline boundary layer depth (meters)
dh     ---> change in boundary layer depth at front or filament (dh <= h0), (meters)

K_bak   --> background vertical diffusivity (m^2 / s)
K0      --> 

b0      --> background buoyancy field (m/s^2)
N2b     -->  background stratification (s^-2)
N20     -->  background minimum stratification in interior (s^-2)
B       --> fractional reduction of surface layer stratification to interior stratificaton
lam_inv --> vertical scale of transition between the two regimes (1/m)


ubar_choice --> 'TW' or 'zero': initialize barotropic longitudinal flow for initial pressure calculation
                                with a TTW flow field (ubar_choice = 'TW') based in b(y,z) and Kv(y,z) or have
                                ubar(y) = 0 with (ubar_choice = 'zero')
'''
###############################
prof_IC = 'FIL'
dom_IC  = 'DW' #DW HERE TO CREATE STRATIFICATION

lh  = 3000
h0  = 15
dh  = 15

K_bak = 1E-4
K0    = 0.02
sig0  = 0.05

b0     = 6.4E-3
N2b    = 1E-6
N20    = 3.4E-5
B      = 0.025
lam_inv = 1./3.

ubar_choice = 'TW'
####################################
# INITIAL CONDITION SURFACE FORCING
'''
Q0_IC     --> initial surface heat flux forcing (W/m^2)
sustr0_IC --> initial longitiduinal wind-stress forcing (m/s^2)
svstr0_IC --> initial transverse wind-stress forcing (m/s^2)
'''
###################################
Q0_IC = 0 
sustr0_IC = 0
svstr0_IC = 0.1 / rho0
###################################
# DIURNAL CYCLE PRESCRIPTION CONSTANTS
# FOR SOLAR OR PRESCRIBED Kv FORCING
'''
alpha_in    --> controls functional form of diurnal cycle forcing 
beta_in     --> controls functional form of diurnal cycle forcing
                e.g., Q(t) = 0.5 * ( alpha_in * P(t) + beta_in * cos(t))
                      where P(t) is a sigmoid form of an oscillation
    
hour_shift --> used for P(t) construction of sigmoid diurnal cycle (irrelevant if alpha_in = 0)
hrs_day    --> used for P(t) construction of sigmoid diurnal cycle (irrelevant if alpha_in = 0)

delta_sustr --> relative change in longitidunal wind stress (m/s^2)
delta_svstr --> relative change in transverse wind stress (m/s^2)

Q0_spin     --> for spinup procedure, magnitude to spin heat flux down to (W/m^2)
Q0          --> where Q(t) starts diurnal cycle (W/m^2)
delta_Q     --> sets magnitude width of diurnal transitions (W/m^2) 
                   if you want Q(t) to be in range [-400, 400], set Q0 = -400, delta_Q = 400 
'''
###################################
hour_shift = 4.
hrs_day = hour_shift / 0.17
alpha_in = 0.
beta_in = 2.


delta_sustr = 0.
delta_svstr = 0.08 / rho0

Q0_spin = -100. 
Q0 = -200.
delta_Q = 200.


