###################################################
# create_forcings.py
'''
SCRIPT TO CREATE SURFACE WIND STRESS
AND SOLAR HEAT FLUX (AND POSSIBLY TIDAL)
FORCING PROFIELES [nt,Ly]

 
DEPENDING ON CHOICES MADE IN
PARAMS FILE, CERTAIN TYPES OF FORCING 
TIME SERIES ARE CREATED HERE

THIS SCRIPT IS CALLED AS execfile('create_forcings.py')
SO IT DOES NOT IMPORT ANY MODULES AS IT IS ONLY
USED WITHIN TTTW_main AND ASSUMES VARIABLES
SUCH AS nt,Ly and base/max values for 
forcing paramters are set in main and params
'''
##################################################

#######################################
# STOCK FUNCTION TO MAKE SIGMOID CURVES
'''
INPUTS
F0 --->  BASE AMPLITUDE
FMAX ---> MAX AMPLITUDE
tvec_sec_curve ---> time vector of curve (in seconds)
shift_time ---> time in seconds of where to put shift
twidth ---> width of shift (increase/decrease)

OUTPUTS
F ---> sigmoid curve 
'''
####################################### 
def make_sigmoid_tseries(F0,FMAX,tvec_sec_curve,shift_time,twidth):
    nt_curve = len(tvec_sec_curve)
    F = np.zeros([nt_curve])
    for t in range(nt_curve):
        F[t] = (F0 / 2) * (1 + np.tanh( (shift_time - tvec_sec_curve[t])  / (0.5*twidth))) + (FMAX - F0)
    return F
    #####################################################################################################

def make_sigmoid_tseries_unity(tvec_sec_curve,shift_time,twidth):
    nt_curve = len(tvec_sec_curve)
    F = np.zeros([nt_curve])
    for t in range(nt_curve):
        F[t] =  (np.tanh( (shift_time - tvec_sec_curve[t])  / (0.5*twidth))) 
    return F
    ############################################################################

def make_composite_day(dt, tvec_sec, hour_shift, alpha=0, beta=2.):
    '''
    CREATE A COMPOSITED VERSION OF A DIURNAL CYCLE
    FOR A SINGLE DAY USING THE FOLLOWING FORMULATION

    D(t) = 0.5 * (alpha * P(t,t_w) + beta * cos(t/T))

    Where P(t,t_w) is a simgoid mirroring time-series and alpha
    and beta are tuning factors for the shape and speed of the composite
    time series

    alpha = 0 is the default case which is also the slowest transition
    
    D(t) is in the range (-1, 1) 
    '''
    # SET UP PARAMETERS FOR P(t,t_w)
    hrs_day = hour_shift / (4./24.)
    len_day_tsteps_hr_shift = len(np.arange(0,hrs_day*60*60,dt))
    if len_day_tsteps_hr_shift %2!=0:
       len_day_tsteps_hr_shift+=1

    len_day_sec = hrs_day * 60 * 60
    delta = 2
    shift_time = (hrs_day * 60 * 60) / (2 * delta)
    T  = len(np.arange(0,24*60*60,dt))
    hour_diff = 4 - hour_shift #4 hour maxmimum shift
    T_d = hour_diff * 60 * 60 / dt

    #CREATE HALF-DAY WITH SIGMOID FUNCTION
    twidth = hour_shift * 60 * 60
    S_1 = make_sigmoid_tseries_unity(tvec_sec[0:len_day_tsteps_hr_shift/delta +1], shift_time,twidth)
    #NORMALIZE S SO THAT INITIAL VALUE = 1
    S = S_1 / S_1[0]

    P_day = np.zeros([T])
    P_day[0:T_d] = S[0]
    T_s = len(S)
    P_day[T_d:T_d + T_s] = S

    if T_d + T_s<=T/2:
       P_day[T_d + T_s:T - (T_d + T_s)] = S[-1]
       P_day[T - (T_d + T_s):T-T_d] = S[::-1]
       P_day[T - T_d::] = S[0]
    else:
       P_day[T_s + T_d::] = S[::-1][0:T-(T_s + T_d)]

    ##########################################
    # CREATE cos(t/T)
    ##########################################
    K_cos = np.zeros([T])
    for t in range(T):
        K_cos[t] = np.cos(t / float(T) * (2 * np.pi))

    ################################################
    # CREATE COMPOSITE
    ################################################
    K_comp = 0.5 * (alpha * P_day + K_cos * beta)

    return K_comp
    #####################################################



print ' 	########################################################'
print '		CREATING TIME SERIES OF FORCINGS WITH CHOICES'
print ' 		wind_tseries: ' + wind_tseries
print ' 		Q_tseries: '     + Q_tseries
print '                 K_tseries: '     + K_tseries
print ' 	#########################################################'

#########################################
#	SURFACE WIND STRESS
'''
BASIC CODE STRUCTURE IS JUST ADD IF/ELSE
STATEMENTS WITH STRING INDICATING TYPE
OF TIME SERIES WANTED

BUT MAIN PARAMTERS ARE 
svstr0 ---> base amplitude wind stress
svstr_max ---> max amplitude wind stress

sustr0 ---> analgous to svstr0
sustr_max --> analagous to svstr_max
'''
#########################################
# DEFAULT IS ZEROS
svstr_nt = np.zeros([nt,Ly+1])
sustr_nt = np.zeros([nt,Ly])

############################
'''
CONSTANT STRESS IN TIME
'''
#############################
if wind_tseries == 'constant':
   svstr_nt[:,:] = svstr0_IC
   sustr_nt[:,:] = sustr0_IC

if wind_tseries == 'Kcomp_diurnal':
   '''
   DIURNAL TIME SERIES OF WIND
   '''
   wind_nt = np.zeros([nt])
   # FIRST MAKE ONE DAY'S WORTH OF A DIURNAL TIME SERIES
   len_day_sec = 24 * 60 * 60
   delta = 2
   shift_time = len_day_sec / (2*delta)
   len_day_tsteps = len(np.arange(0,24*60*60,dt))
   #hour_shift = 4 SHOULD BE IN PARAMS
   twidth = hour_shift * 60 * 60

   #######################################
   #CREATE COMPOSITE OF DAY DIURNAL CYCLE
   ########################################
   Kcomp_tseries = 1 + make_composite_day(dt, tvec_sec, hour_shift, alpha=alpha_in, beta=beta_in)
   sustr_diurnal_day = (delta_sustr / 2.) * Kcomp_tseries + sustr0_IC - delta_sustr 
   svstr_diurnal_day = (delta_svstr / 2.) * Kcomp_tseries + svstr0_IC - delta_svstr 

   #######################################
   # APPLY SPINUP
   ########################################
   #HOLD K-CONSTANT FOR PERIOD OF DAYS TO ALLOW EQUILLIBARTION
   const_days = 3 
   tsteps_const = const_days * len_day_tsteps 
   su_temp_nt = np.zeros([nt])
   sv_temp_nt = np.zeros([nt])
   su_temp_nt[0:tsteps_const] = sustr_diurnal_day[0]
   sv_temp_nt[0:tsteps_const] = svstr_diurnal_day[0]


   tind = tsteps_const 
   for d in range(tend_days-const_days):
       su_temp_nt[tind:tind+len_day_tsteps] = sustr_diurnal_day[:]
       sv_temp_nt[tind:tind+len_day_tsteps] = svstr_diurnal_day[:]
       tind = tind + len_day_tsteps


   for j in range(Ly):
       sustr_nt[:,j] = su_temp_nt
       svstr_nt[:,j] = sv_temp_nt
    ###############################################################

##########################################
'''
Start out at constant wind
and linearly spin down to near zero value
and hold constant for rest of simulation
'''
########################################
if wind_tseries == 'V1_spindown_tozero':
   len_day_tsteps = len(np.arange(0,24*60*60,dt))
   total_spinup_days = 2
   tstep_spinup = total_spinup_days * len_day_tsteps 

   su_temp_nt = np.zeros([nt])
   sv_temp_nt = np.zeros([nt])
   
   # 1ST DAY CONSTANT WIND  
   spinup_const_tsteps = 1 * len_day_tsteps
   
   su_temp_nt[0:spinup_const_tsteps] = sustr0_IC
   sv_temp_nt[0:spinup_const_tsteps] = svstr0_IC
  
   # 2ND DAY SPIN DOWN
   #su_spin_to = sustr0_IC - delta_sustr *0.5
   #sv_spin_to = svstr0_IC - delta_svstr *0.5
   for t in range(len_day_tsteps,len_day_tsteps*2):
       su_temp_nt[t] = (sustr0_IC - (delta_sustr/2. * (( tvec_sec[t-len_day_tsteps]/(86400/2))))) 
       sv_temp_nt[t] =  (svstr0_IC - (delta_svstr/2. * (( tvec_sec[t-len_day_tsteps]/(86400/2))))) 


   su_temp_nt[len_day_tsteps*2::] = su_temp_nt[2*len_day_tsteps-1] 
   sv_temp_nt[len_day_tsteps*2::] = sv_temp_nt[2*len_day_tsteps-1] 
  
   for j in range(Ly):
       sustr_nt[:,j] = su_temp_nt
       svstr_nt[:,j] = sv_temp_nt
   svstr_nt[:,-1] = sv_temp_nt

   ####################################



#########################################
# 		SOLAR HEAT FLUX 
'''
Q0 ---> BASE AMPLITUDE FOR HEAT FLUX
'''
#########################################
Q_nt = np.zeros([nt,Ly])

#############################
'''
Diurnal cycle of Q
with a spinup from Q=0 to 
a linear increase/decrease in
Q to an eventual diurnal cycle

K_comp --> sigmoid + cos composite form
'''
############################

if Q_tseries == 'Kcomp_diurnal_linear_spin':
     # FIRST MAKE ONE DAY'S WORTH OF A DIURNAL TIME SERIES
     len_day_sec = 24 * 60 * 60
     delta = 2
     shift_time = len_day_sec / (2*delta)
     len_day_tsteps = len(np.arange(0,24*60*60,dt))
     twidth = hour_shift * 60 * 60

     #######################################
     #CREATE COMPOSITE OF DAY DIURNAL CYCLE
     ########################################
     Q_diurnal_day =-delta_Q*0.5* make_composite_day(dt, tvec_sec, hour_shift, alpha=alpha_in, beta=beta_in)

     #######################################
     # APPLY SPINUP
     ########################################
     total_spinup_days = 2
     
     tsteps_spinup = total_spinup_days * len_day_tsteps
     Q_nt[0:tsteps_spinup/total_spinup_days,:] = 0
     # SPINUP TIME IS LINEAR DECREASE IN Q 
     spinup_days_Q_linear = 1 
     tsteps_spinup_end = (spinup_days_Q_linear * len_day_tsteps) + tsteps_spinup/total_spinup_days
     #Q0_spin = -300 in params
     for t in range(tsteps_spinup/total_spinup_days,tsteps_spinup_end+1):
         Q_nt[t,:] = Q0_spin*( (tvec_sec[t-(tsteps_spinup/total_spinup_days)])/86400 )

     #HOLD Q-CONSTANT FOR PERIOD OF DAYS TO ALLOW EQUILLIBARTION
     const_days = 2
     tsteps_const = const_days * len_day_tsteps 
     tind = tsteps_spinup
     Q_nt[tind:tind+tsteps_const,:] = Q_nt[t,:]

     tind = tind + tsteps_const 
     

     for d in range(tend_days-(total_spinup_days+const_days)):
         for j in range(Ly):
             Q_nt[tind:tind+len_day_tsteps,j] = Q_diurnal_day[:]
         tind = tind + len_day_tsteps

     ###############################################################


################################
# PRESCRIBED VERTICAL MINXG
###############################
if K_tseries =='constant':
   Kv_tseries[:,:,:] = K0
   Kt_tseries[:,:,:] = K0





















