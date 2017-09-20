#####################################
# TTTW_kpp.py
'''
KPP paramterization implentend for
TTTW system (z-level vertical coordinate system,
model domain dimensions)

Based off of ROMS working group
KPP July 2015
'''

# DANIEL DAUHAJRE, UCLA JULY 2015
#####################################


######################################
import os
import sys
import numpy as np
import TTTW_funcs as TTTW_func
from scipy import signal as scipysig
from scipy import ndimage
#####################################




class TTTW_KPP(object):
    

    def __init__(self,b,u,v,hbls_old,hbbl_old,Kv_old,Kt_old,srflx,sustr,svstr,f,grid_dict,tstep_mode,dt):
        """ Initialize KPP object with inputs from TTTW system"""
       
        # INPUTS FROM TTTW SYSTEM
        self.b       = b #buoyancy field: [Ly,N]
        self.u       = u # x-component of velocity [Ly,N]
        self.v       = v # y-component of velocity [Ly+1,N]
        self.hbls_old = hbls_old #boundary layer depth from previous time step [Ly]
        self.hbbl_old = hbbl_old # bottom boundary layer depth from previous time step [Ly]
        self.Kv_old   = Kv_old # momentum mixing coefficeint from previous time step [Ly,N+1]
        self.Kt_old   = Kt_old # tracer mixing coefficient from previous time step [Ly,N+1]
        self.srflx = srflx #solar heat flux [Ly] (degC * (m/s))
        self.sustr = sustr # x-component surface wind stress [Ly] (N/m^2) 
        self.svstr = svstr # y-component surface wind stress [Ly+1] (N/m^2)
        self.grid_dict = grid_dict #gridded data
        self.f         = f #coriolis parameter
        # KPP-SPECIFIC VARIABLES  
        self.hbls = np.zeros([self.b.shape[0]])
        self.hbbl = np.zeros([self.b.shape[0]])
        self.ustar = []
        self.bvf = [] 
        self.kmo = []
        self.C_h_MO = []
        self.kbl    = []
        self.Cr     = []      
        self.Fc     = []
        self.ghat   = [] #NONLOCAL TERM: TO BE USED IN TIME STEPPING
        self.tstep_mode = tstep_mode# if in time steppign mode, turn on HBL_RATE_LIMIT
        self.dt = dt

    #########################################################################
    # ONLY FUNCTION NEEDED TO BE CALLED IN TTTW MAIN OR TIME STEPPING SCRIPT
    ########################################################################
    def run_KPP(self):
        # SET SWITCHES
        self.set_switches()
        # SET CONSTANTS
        self.set_constants()
        # CALCULATE SURFACE BUOYANCY FLUXES 
        self.Bo_Bosol_calc()
        #CALCULATE friction velocity
        self.ustar_calc()
        # CALCULATE BVF
        self.bvf_calc()    
        # GET hbls and hbbl
        self.get_hbls_hbbl()
        self.hbls_check1 = np.copy(self.hbls)


        if self.SMOOTH_HBL:
           #self.hbls = self.gauss_smooth_1D(10,self.hbls)
           self.hbls = self.ndimage_smooth(self.hbls,self.sigma_gauss)
           if self.LMD_BKPP:
              #self.hbbl = self.gauss_smooth_1D(10,self.hbls)
              self.hbbl = self.ndimage_smooth(self.hbbl,self.sigma_gauss)

	# RATE LIMIT
	if self.tstep_mode:
           if self.HBL_RATE_LIMIT:
              self.hbl_rate_limit()
        # MERGE OVERWRAP
        if self.MERGED_OVERWRAP and self.LMD_KPP and self.LMD_BKPP:
           self.merge_overwrap()

        # SURFACE KPP 
        if self.LMD_KPP:
           # FIND NEW BOUNDARY LAYER INDEX kbl
           self.find_new_kbl()
           # CALCULATE TURBULENT VELOCITY SCALES AND SHAPE FUNCTIONS
           self.get_bforce_wm_ws_Gx_surf()
           #CALCULATE SURFACE KPP MIXING COEFFICIENTS
           self.compute_mixing_coefficients_surf()

        # BOTTOM KPP 
        if self.LMD_BKPP:
          # CALCULATE TURBULENT VELOCITY SCALES AND SHAPE FUNCTIONS
           self.get_wm_ws_Gx_bot()
           #CALCULATE BOTTOM KPP MIXING COEFFICIENTS
           self.compute_mixing_coefficients_bot()

        #FINALZE MIXING COEFFICIENTS TO USE IN TIME STEPPING 
        self.finalize_Kv_Kt()

        if self.SMOOTH_K:
           self.smooth_Kv_Kt()

    ##################################################################################
    # SWITCHES FOR KPP --> HARDCODED IN HERE, USER SETS TO TRUE OR FALSE BEFORE USING
    # THE MODULE
    # TRUE --> SWITCH IS ON
    # FALSE --> SWITCH IS OFF
    ###################################################################################
    def set_switches(self):
        self.LMD_KPP             = True
        self.LMD_BKPP            = True
        self.LIMIT_MO_DEPTH      = False 
        self.SALINITY            = False
        self.SMOOTH_HBL          = True
        self.SMOOTH_K            = False
        self.MERGED_OVERWRAP     = True 
        self.LIMIT_EKMAN_OLD_WAY = False
        self.LIMIT_UNSTABLE_ONLY = False
        self.LMD_NONLOCAL        = True
        self.MLCONVEC            = False 
        self.LMD_CONVEC          = False
        self.BBL_CONVEC          = False
        self.CD_SWITCH           = True 
	if self.tstep_mode:
	   self.HBL_RATE_LIMIT      = False

    def set_constants(self):
        """ SET CONSTANT PARAMTERS """

        # GLOBAL CONSTANTS (non-specific to FORTRAN lmd_kpp.F code)
        self.alpha = 2E-4 #thermal expansion coefficient
        self.Cp    = 3985 #specific heat of water
        self.rho0  = 1027.5 # reference density
        self.g     = 9.8 

        # Constants set in lmd_kpp.F
        self.nubl   = 0.01 #maximum allowed boundary layer
        self.nu0c   = 0.1 # convective adjustment for viscosity and 
                    #                     diffusivity [m^2/s]

        self.ffac   = 2000. #multiplicative factor for nu0c in CA in ML

        self.Cv     = 1.8  # ratio of interior Brunt-Vaisala Frequency
                    # "N" to that at entrainment depth "he"
        self.Ricr   = 0.45 # Critical bulk richardson number
        self.Ri_inv = 1./self.Ricr
        self.betaT  = -0.2 # ratio of entrainment flux to surface
      		     #      buoyancy flux       
        self.epssfcs = 0.1 # nondimensional extent of the surface layer
        self.epssfcb = 0.084 #nondimensional extent of the bottom layer
        self.C_MO   = 1.   # constant for computiation of Monin-Obukhov depth
        self.C_Ek   = 258. # constant for computing stabilzation term 
                      # due to rotation (Ekman depth limit)
        self.Cstar  = 10.  # proportionality coefficient paramaterizing
                      #     	nonlocal transport
        self.eps     = 1E-20

        self.V0      = 0.15 #hbl depth overrride: Ist+Isrot + V0**2/d 
                            # or used in augmented Vtsq condition
    
        self.V0 = 0. 
        self.vonKar = 0.41
        ######################################
        # Maximum stability parameters  " zeta"
        # value of the 1/3 power law regime of
        # of flux profile for momentum and tracers
        # and coefficents of flux profile for
        # momentum and tracers in their 1/3 power
        # law regime
        #####################################
        self.zeta_m  = -0.2  
        self.a_m     = 1.257
        self.c_m     = 8.360
        self.zeta_s  = -1.  
        self.a_s     = 28.86
        self.c_s     = 98.96   
        self.r2 = 0.5
        self.r3 = 1./3.
        self.r4 = 0.25
       
        self.nu0ml = 0.5 # mixed layer convective adjustment for diffusivty
       
        # testing for iterative solution
        #self.my_Kv_bak = 1E-3
        #self.my_Kt_bak = 1E-3
 
        # OLD VALUES
        #self.my_Kv_bak = 1E-6 #background viscosity/diffusivity
        #self.my_Kt_bak = 1E-5 # background viscosity/diffusivity
       
        self.my_Kv_bak = 1E-4
        self.my_Kt_bak = 1E-4
    

        # TESTING
	#self.Zob = 0.55
        self.Zob =1E-2 
        #self.Zob = 1E-8 #testing r_D values
        #self.C_D = 1E-2 # bulk drag coefficient        


        # PARAMETER FOR GAUSSIAN SMOOTHING OF FIELDS
        self.sigma_gauss = 3
         
	self.C_D = TTTW_func.calc_CD(self.Zob,self.grid_dict)
        self.Cg = self.Cstar * self.vonKar * (self.c_s*self.vonKar * self.epssfcs) **(1./3.)
        self.Vtc = self.Cv * np.sqrt(-self.betaT/(self.c_s*self.epssfcs)) / (self.Ricr*self.vonKar**2) 


        #self.V0 = 0.1 #corrective term for Vtsq
   

    ###############################################
    # FUNCTIONS TO CALCULATE SOME TERMS FOR
    # FOR BL DEPTH DETERMINIATION
    #############################################
    def ustar_calc(self):
        """ Calculate friction velocity (ustar) """
        svstr_upts = TTTW_func.v2u(self.svstr)
        self.ustar = np.sqrt(np.sqrt(self.sustr**2 + svstr_upts**2))
        
    def bvf_calc(self):
        """ Calculate Brunt-Vaisall Frequency"""
        b = self.b
        rho = -self.b *self.rho0 / self.g
        z_u_w = self.grid_dict['z_u_w'][:,:]
        z_u_r = self.grid_dict['z_u_r'][:,:]
        [Ly,N] = b.shape

        # BVF AT W-levels (according to ROMS)
        # but top and bottom just same bvf[:,-2] and bvf[:,1] respectively
        self.bvf = np.zeros([Ly,N+1])
        rho_atw = TTTW_func.rho2w(rho)
        for k in range(N-1):
            k_w = k +1
            self.bvf[:,k_w] = (-self.g / (self.rho0)) * ( ( rho[:,k+1] - rho[:,k]) / (z_u_r[:,k+1] - z_u_r[:,k]) )
         
        # TAKE CARE OF TOP AND BOTTOM
        self.bvf[:,0] = self.bvf[:,1]
        self.bvf[:,-1] = self.bvf[:,-2]
        
    def Bo_Bosol_calc(self):
        """ Get surface turbulent buoyancy forcing (Bo and Bosol)"""
        self.Bosol = (self.g*self.alpha * self.srflx ) 
        #ZEROS FOR T3W APPLICATION
        self.Bo = np.zeros([self.b.shape[0]])
   


    def lmd_wscale_ws_only(self,Bfsfc,zscale,hbl,ustar):
        if self.LIMIT_UNSTABLE_ONLY:
           if Bfsfc < 0:
              zscale = np.min([zscale, hbl*self.epssfcs])
        else:
            zscale = np.min([ zscale,hbl * self.epssfcs])
           
        zetahat = self.vonKar * zscale * Bfsfc
        ustar3  = ustar**3
        
        # STABLE REGIME
        if zetahat >= 0:
           ws = self.vonKar * ustar * ustar3 / np.max([ustar3+5.*zetahat,1E-20])
        # UNSTABLE REGIME
        elif zetahat > self.zeta_s * ustar3:
             ws = self.vonKar * (( ustar3-16.*zetahat)/ustar) **self.r2
        # CONVECTIVE REGIME
        else:
            ws = self.vonKar * (self.a_s*ustar3-self.c_s*zetahat)**self.r3
        
        return ws   


    ##########################################################
    # FUNCTION TO OBTAIN MIXED LAYER DEPTHS (hbls and hbbl)
    ###########################################################
    def get_hbls_hbbl(self):
        """ Determine depth of surface and bottom boundary layers"""
        [Ly,N] = self.b.shape
        z_u_w  = self.grid_dict['z_u_w']
        z_u_r  = self.grid_dict['z_u_r']
        u      = self.u
        v      = self.v
       
        v_upts = TTTW_func.v2u(v)
        Hz = z_u_w[:,1:] - z_u_w[:,:-1]



        # CALCULATE swr_frac
        self.swr_frac = TTTW_func.lmd_swr_frac(self.grid_dict)


        # WHOLE THING HAPPENS IN j loop through y-indices
        
        # INITIALIZE ARRAYS
        self.kmo    = np.zeros([Ly])
        self.Cr     = np.zeros([Ly])
        self.kbl    = np.empty([Ly],dtype='int')
        self.C_h_MO = np.zeros([Ly])
        self.Cr     = np.zeros([Ly,N+1]) # sum term
        self.FC     = np.zeros([Ly,N+1])
        self.swdk_r = np.zeros([Ly,N+1])
 
        #DEBUGGING Cr
        #self.out1 = np.zeros([Ly,N+1])
        #self.out2 = np.zeros([Ly,N+1])
        #self.out3 = np.zeros([Ly,N+1])
        #self.out4 = np.zeros([Ly,N+1])
        self.zscale = np.zeros([Ly,N])
        self.Kern = np.zeros([Ly,N])

 
        # --> LOOP THROUGH Y-INDICES
        for j in range(Ly):
            if self.LIMIT_MO_DEPTH:
               self.kmo[j] = 0
               self.C_h_MO[j] = self.C_MO *self.ustar[j]**3/self.vonKar
            
            self.kbl[j] = 0
            self.Cr[j,-1] = 0 # set top Cr
            self.Cr[j,0]  = 0 # set bottom Cr
            
            # SEARCH FOR MIXED LAYER DEPTH
            self.FC[j,-1] = 0.


            #self.out1[j,-1] = 0
            #self.out2[j,-1] = 0
            #self.out3[j,-1] = 0                    
            

            # ---> LOOP TOP TO BOTTOM (FORTRAN ==> k=N-1,1,-1)
            for k in range(N-1,0,-1):
                # INDEX MAP
                k_r = k-1
                k_w   = k

   
                zscale = z_u_w[j,N] - z_u_r[j,k_r]
                self.zscale[j,k_w] = zscale
                if self.LMD_KPP:
                   if self.LMD_BKPP:
                       zscaleb = z_u_r[j,k_r] - z_u_w[j,0]
                       Kern = zscale * zscaleb**2 / ( (zscale + self.epssfcs*self.hbls_old[j]) * (zscaleb**2+(self.epssfcb**2*self.hbbl_old[j]**2)))
                   else:
                       Kern   = zscale / (zscale + (self.epssfcs*self.hbls_old[j]))
                else:
                   Kern = 1.
                


                self.Kern[j,k_w] = Kern
                # in lmd_kpp.F they go to rho-points in this
                # not necessary here, u at rho-points and v_upts defined above
                 
                #self.out1[j,k_w] = self.out1[j,k_w+1] + Kern * (( (v_upts[j,k_r+1] - v_upts[j,k_r])**2) / (Hz[j,k_r] + Hz[j,k_r+1]))


                #self.out1[j,k_w] = self.out1[j,k_w+1] + Kern * ( ((u[j,k_r+1] - u[j,k_r])**2 + (v_upts[j,k_r+1] - v_upts[j,k_r])**2) / (Hz[j,k_r] + Hz[j,k_r+1]))

                #self.out2[j,k_w] = self.out2[j,k_w+1] + Kern * ( -0.5 * (Hz[j,k_r]+Hz[j,k_r+1]) * (self.Ri_inv * self.bvf[j,k_w]))
                #self.out3[j,k_w] = self.out3[j,k_w+1] + Kern * (-0.5 * (Hz[j,k_r] + Hz[j,k_r+1]) * (self.C_Ek * self.f[j] * self.f[j]))

                self.FC[j,k_w] = self.FC[j,k_w+1] + Kern * (\
                                     ( ( u[j,k_r+1] - u[j,k_r] )**2 + ( v_upts[j,k_r+1] - v_upts[j,k_r])**2 ) \
                                     / (Hz[j,k_r] + Hz[j,k_r+1]) \
                                     - 0.5 * ( Hz[j,k_r] + Hz[j,k_r+1]) * (self.Ri_inv * self.bvf[j,k_w] + self.C_Ek*self.f[j]*self.f[j]))


            #		LOOP THAT FINDS BL DEPTH ##
            #----> LOOP TOP TO BOTTOM (start at free surface, w-level surface) 
            
            if self.LMD_KPP:
                     #swdk_r only used in this function so don't need to be class attribute
                     # but for testing make it an attribute to see what it is
                                                                 
                     # fortran equivlanet ===> k=N,1,-1                   
                     for k in range(N,0,-1):
                     #for k in range((N+1)-1,-1,-1):
                         # INDEX MAP
                         k_r = k-1
                         k_w = k

                         ###################################################################### 
                         self.swdk_r[j,k_w] = np.sqrt( self.swr_frac[j,k_w] * self.swr_frac[j,k_w-1])
                         zscale = z_u_w[j,N] - z_u_r[j,k_r]
                         Bfsfc = self.Bo[j] + self.Bosol[j] * (1-self.swdk_r[j,k_w])
                       
                         self.bvf_max = np.sqrt(np.max([0,self.bvf[j,k_w-1]]))
                         
                         # CALCULATE TURBULENT VELOCITY SCALE FOR TRACERS
     			 self.ws = self.lmd_wscale_ws_only(Bfsfc, zscale,self.hbls_old[j],self.ustar[j])
                           
                         self.Vtsq = self.Vtc * self.ws* self.bvf_max + self.V0
                         #self.out4[j,k_w] = self.Vtsq #debugging
                        

                         self.Cr[j,k_w] = self.FC[j,k_w] + self.Vtsq
                        

                         #######################################################################
                         
                         # SEARCH FOR hbls vertical level #
                         '''
                         kbl is specified at vertical w-level (via Cr which is at
                         vertical w-levels)
                         '''
                         if self.kbl[j] == 0 and self.Cr[j,k_w] < 0:
                            self.kbl[j] = k_w
                         if self.LIMIT_MO_DEPTH:
                            if self.kmo[j] == 0 and Bfsfc*(z_u_w[j,N] - z_u_r[j,k_r]) > self.C_h_MO[j]:
                               self.kmo[j] = k_w

                         
            #--> still in j-loop
            #######################################################
         
            # 		GET SURFACE BOUNDARY LAYER DEPTH #  
            self.hbls[j] = z_u_w[j,N] - z_u_w[j,0] + self.eps # set hbls as depth of entire water column
            if self.kbl[j] > 0:
               k_w = self.kbl[j]
               k_r = k_w - 1              
               if k_w == N: # set hbls at the surface btwn w- and rho-levels at surface
                  self.hbls[j] = z_u_w[j,N] - z_u_r[j,N-1]
 
               else:
                   self.hbls[j] = z_u_w[j,N] - ( z_u_r[j,k_r] * self.Cr[j,k_w+1]  - z_u_r[j,k_r+1] * self.Cr[j,k_w]) / \
                                                                  (self.Cr[j,k_w+1] - self.Cr[j,k_w])
            
            if self.LIMIT_MO_DEPTH:
               if self.kmo[j] > 0:
                  k_w = self.kmo[j]
                  k_r = k_w-1
                  if k_w == N:
                     z_up   = z_u_w[j,N]
                     cff_up = np.max([0,Bo[j]])
                  else:
                      z_up   = z_r[j,k_w+1]
                      cff_up = np.max([0, Bo[j] + self.Bosol[j]*(1-self.swdk_r[j,(k_w-1)+1])])
                   
                  cff_dn = np.max([0,Bo[j] + self.Bosol[j] * (1-self.swdk_r[j,k_w])]) 
                  h_MO   = z_u_w[j,N] + self.C_h_MO[j] * ( cff_up*z_up - cff_dn * z_u_r[j,k_r] ) \
                           / ( cff_up * cff_dn * (z_up - z_u_r[j,k_r]) ) \
                           + self.C_h_MO[j] * (cff_dn - cff_up)

                  self.hbls[j] = np.min([self.hbls[j],np.max([h_MO,0])])



            #### GET BOTTOM BOUNDARY LAYER DEPTH #######
            if self.LMD_BKPP:
               self.kbl[j] = 0 # reset Cr at bottom and kbl for BKPP
               self.Cr[j,0] = 0.
               self.FC[j,0] = 1.5 * self.FC[j,1] - 0.5 * self.FC[j,2] # linear extrapolation
               
               #---> LOOP BOTTOM TO TOP
               # FIND kbl for BBL
               for k in range(1,N+1):
                   k_r = k-1
                   k_w = k 
                   self.Cr[j,k_w] = self.FC[j,k_w] - self.FC[j,0]
                   
                   # LOOK FOR FIRST ZERO CROSSING FROM BOTTOM UP
                   if self.kbl[j] == 0 and self.Cr[j,k_w] > 0:
                      self.kbl[j] = k_w 
                

               self.hbbl[j] = z_u_w[j,N] - z_u_w[j,0] # total depth
               if self.kbl[j] > 0 :
                  k_w = self.kbl[j] 
                  k_r = k_w -1
                  if k_w == 1: # NO BBL CASE
                     self.hbbl[j] = z_u_r[j,0]  - z_u_w[j,0]  #in between bottom rho and w-level
                  else:
                      self.hbbl[j] = ( z_u_r[j,k_r-1] * self.Cr[j,k_w] - z_u_r[j,k_r] * self.Cr[j,k_w-1]) / \
                                     (self.Cr[j,k_w] - self.Cr[j,k_w-1]) - z_u_w[j,0]




    ########################################################################################################## 
    def hbl_rate_limit(self):
	""" APPLY RATE LIMITER TO hbls, and hbbl"""
        tstar =4E4 
        [Ly,N] = self.b.shape
        print 'HBL RATE LIMITER CALLED....'
	print 'tstar = ' + str(tstar) + ' seconds'
	for j in range(Ly):
            dh_s = self.hbls[j] - self.hbls_old[j]
	    if abs(dh_s) >= self.dt / tstar * self.hbls_old[j]:
	       #print ' SBL RATE LIMIT BEING APPLIED at j = ' + str(j) 
               #print ' dh_s = ' + str(dh_s) + 'm'
	       dhm_s = self.dt / tstar * self.hbls_old[j]
	       self.hbls[j] = self.hbls_old[j] + dhm_s*dh_s/abs(dh_s)
	    if self.LMD_BKPP:
	       dh_b = self.hbbl[j] - self.hbbl_old[j]
               if abs(dh_b) >= self.dt / tstar * self.hbbl_old[j]:
                   dhm_b = self.dt / tstar * self.hbbl_old[j]
	           self.hbbl[j] = self.hbbl_old[j] + dhm_b*dh_b/abs(dh_b)

    def merge_overwrap(self):
        """ If hbls and hbbl overlap then replace them with total depth """
        [Ly,N] = self.b.shape
        z_u_w = self.grid_dict['z_u_w']
        z_u_r = self.grid_dict['z_u_r']
        for j in range(Ly):
            cff = z_u_w[j,N] - z_u_w[j,0]
            if self.hbls[j] + self.hbbl[j] > cff:
               self.hbls[j] = cff
               self.hbbl[j] = cff

    def smooth_Kv_Kt(self):
        for k in range(self.Kv_out.shape[1]):
            self.Kv_out[:,k] = self.ndimage_smooth(self.Kv_out[:,k],self.sigma_gauss)
            self.Kt_out[:,k] = self.ndimage_smooth(self.Kt_out[:,k],self.sigma_gauss)

    def ndimage_smooth(self,field,sigma):
        return ndimage.gaussian_filter(field,sigma)


    def find_new_kbl(self):
        """ After hbls and hbbl are found, find the new BL vertical index to use in constructing Kv field """
        [Ly,N] = self.b.shape
        z_u_w  = self.grid_dict['z_u_w']
        z_u_r  = self.grid_dict['z_u_r']

        #---> j loop
        for j in range(Ly):
            self.kbl[j] = N #initialize search at top

        # in fortran k=N-1,1,-1
        for k in range(N-1,0,-1):
            #INDEX MAP
            k_w = k
            k_r = k-1
   
            for j in range(Ly):
                if z_u_w[j,k_w] > z_u_w[j,N] - self.hbls[j]:
                   self.kbl[j] = k_w


    def lmd_wscale_wm_and_ws(self,Bfsfc,zscale,ustarj,hblsj):
        """" Calculate turbulent velocity scales for momentum and scalars"""
        if self.LIMIT_UNSTABLE_ONLY:
           if Bfsfc < 0:
              zscale = np.min([ zscale,hblsj*self.epssfcs])
        else:
            zscale = np.min([zscale,hblsj*self.epssfcs])

        zetahat = self.vonKar * zscale * Bfsfc
        ustar3  = ustarj**3
       
        # STABLE REGIME
        if zetahat>=0:
           wm = self.vonKar * ustarj * ustar3/np.max([ustar3+5.*zetahat,1E-20])
           ws = wm
        #UNSTABLE REGIME
        else:
            if zetahat > self.zeta_m * ustar3:
               wm = self.vonKar * (ustarj*(ustar3-16*zetahat)) **self.r4
            else:
               wm = self.vonKar * (self.a_m*ustar3-self.c_m*zetahat)**self.r3
            
            if zetahat > self.zeta_s * ustar3:
               ws = self.vonKar * ( ( ustar3 - 16.*zetahat)/ustarj)**self.r2
            else:
               ws = self.vonKar*(self.a_s * ustar3 - self.c_s*zetahat) ** self.r3

        return wm,ws
               


    def get_bforce_wm_ws_Gx_surf(self):
        """ Find buoyancy forcing for final hbl values and 
            compute turbulent velocity scales (wm,ws) at hbl
            and compute nondimensional shape function coefficeints Gx()
            by matching values and vertical derivatives of interior mixing coefficients
            at hbl (sigma = 1) """

        [Ly,N] = self.b.shape
        z_u_w = self.grid_dict['z_u_w']        
    
        self.Gm1     = np.zeros([Ly])
        self.dGm1_dS = np.zeros([Ly]) 
        self.Gt1     = np.zeros([Ly])
        self.dGt1_dS = np.zeros([Ly]) 
        self.Bfsfc_bl = np.zeros([Ly])
        self.Av_bl = np.zeros([Ly])
        self.dAv_bl = np.zeros([Ly])
       
        #debugging
        self.wm_surf = np.zeros([Ly])
        self.ws_surf = np.zeros([Ly]) 

        #---> j-loop
        for j in range(Ly): 
            k_w    = self.kbl[j] # KBL is "new bl index after calling find_new_kbl()
            z_bl   = z_u_w[j,N] - self.hbls[j]
            zscale = self.hbls[j] 
            
            if self.swr_frac[j,k_w-1] > 0:
               Bfsfc = self.Bo[j] + self.Bosol[j] * ( 1. - self.swr_frac[j,k_w-1]\
                                                  * self.swr_frac[j,k_w] * ( z_u_w[j,k_w] - z_u_w[j,k_w-1] )\
                                                  / (self.swr_frac[j,k_w] * (z_u_w[j,k_w] - z_bl)\
                                                  + self.swr_frac[j,k_w-1] * (z_bl - z_u_w[j,k_w-1]) ))
           
            else:
                Bfsfc = self.Bo[j] + self.Bosol[j]
   
            # CALCUALTE TURBULENT VELOCITY SCALES
            wm,ws = self.lmd_wscale_wm_and_ws(Bfsfc,zscale,self.ustar[j],self.hbls[j])
            self.wm_surf[j]  = wm
            self.ws_surf[j] = ws            

            if self.LIMIT_UNSTABLE_ONLY:
               f1 = 5. * np.max([0,Bfsfc]) * self.vonKar / (self.ustar[j]**4+self.eps)
            else:
               f1 = 0

            
            cff    = 1. / (z_u_w[j,k_w] - z_u_w[j,k_w-1])
            cff_up = cff * (z_bl - z_u_w[j,k_w])
            cff_dn = cff * (z_u_w[j,k_w] - z_bl)

            #MOMENTUM 
            Av_bl      = cff_up * self.Kv_old[j,k_w] + cff_dn * self.Kv_old[j,k_w-1]
            dAv_bl     = cff * (self.Kv_old[j,k_w] - self.Kv_old[j,k_w-1])
            self.Av_bl[j] = Av_bl
            self.dAv_bl[j] = dAv_bl
            self.Gm1[j]     = Av_bl / (self.hbls[j] * wm + self.eps)
            self.dGm1_dS[j] = np.min([0.,Av_bl*f1-dAv_bl/(wm+self.eps)])  

            #TEMPERATURE(BUOYANCY)
            At_bl      = cff_up * self.Kt_old[j,k_w] + cff_dn * self.Kt_old[j,k_w-1]
            dAt_bl     = cff * (self.Kt_old[j,k_w] - self.Kt_old[j,k_w-1])
            self.Gt1[j]     = At_bl / (self.hbls[j] * ws + self.eps)
            self.dGt1_dS[j] = np.min([0.,At_bl*f1-dAt_bl/(ws+self.eps)])  

            self.Bfsfc_bl[j] = Bfsfc



    def get_wm_ws_Gx_bot(self):
        """ Compute turbulent velocity scales and Gx for bottom kpp"""
            # BASICALLY SETS self.Gm1_bot, self.dGm1_dS_bot, self.Gt1_bot, self.dGt1_dS_bot 
        z_u_r = self.grid_dict['z_u_r']
        z_u_w = self.grid_dict['z_u_w']
        [Ly,N] = self.b.shape
        #---> j-loop
        for j in range(Ly):  
            self.kbl[j] = N # initialize search
        #-> end j-loop

        #--> k-loop
        for k in range(N-1,0,-1):
            k_w = k
            k_r = k-1
            # --> j loop 
            for j in range(Ly):
                if z_u_r[j,k_r] - z_u_w[j,0] > self.hbbl[j]:
                   self.kbl[j] = k_w

            #--> end k
        # --> end j


        '''
        Compute nondimenisonal shape function coefficeints Gx() by
        matching values and vertical derivatives of interior mixing
        coefficients at hbbl (sigma=1)
        '''

        self.Gm1_bot     = np.zeros([Ly])
        self.dGm1_dS_bot = np.zeros([Ly])
        self.Gt1_bot     = np.zeros([Ly])
        self.dGt1_dS_bot = np.zeros([Ly]) 
        self.Av_bl_bot     = np.zeros([Ly])
        self.dAv_bl_bot = np.zeros([Ly]) 
        self.cff_up_bot = np.zeros([Ly])
        self.cff_dn_bot = np.zeros([Ly])





        self.wm_bot = np.zeros([Ly])
        self.ws_bot = np.zeros([Ly])        

        # CALCULATE ustar for the bottom based on bototm velocities
        
   
        
        # CALCULATE r_D
        self.r_D = TTTW_func.get_r_D(self.u,self.v,self.Zob,self.grid_dict) 
        u = self.u
        v_upts = TTTW_func.v2u(self.v)
        
        ubar = np.mean(u,axis=1)
        vbar = np.mean(v_upts,axis=1)

        # --> j loop
        for j in range(Ly):
            # turbulent velocity sclaes with buoyancy effects neglected
            if self.CD_SWITCH:
               # DEPTH AVERAGED APPROACH
               uref =  u[j,0]
               vref =  v_upts[j,0]
               ustar2 = self.C_D * (uref**2 + vref**2)
            else:
               ustar2 = self.r_D[j] * np.sqrt(u[j,0]**2 + v_upts[j,0]**2)
            wm = self.vonKar * np.sqrt(ustar2)
            ws = wm

            self.wm_bot[j] = wm
            self.ws_bot[j] = ws
 
            k_w = self.kbl[j] 
            z_bl = z_u_w[j,0] + self.hbbl[j]

            if z_bl < z_u_w[j,k_w-1]:
               k_w = k_w-1

            cff    = 1. / (z_u_w[j,k_w] - z_u_w[j,k_w-1])
            cff_up = cff * (z_bl - z_u_w[j,k_w])
            cff_dn = cff * (z_u_w[j,k_w] - z_bl)
          
            Av_bl               = cff_up * self.Kv_old[j,k_w] + cff_dn * self.Kv_old[j,k_w-1]
            dAv_bl              = cff * ( self.Kv_old[j,k_w] - self.Kv_old[j,k_w-1])
            self.Av_bl_bot[j] = Av_bl
            self.dAv_bl_bot[j] = dAv_bl


            self.Gm1_bot[j]     = Av_bl / (self.hbbl[j] * wm + self.eps)
            self.dGm1_dS_bot[j] = np.min([0,-dAv_bl/(ws+self.eps)])

            At_bl               = cff_up * self.Kt_old[j,k_w] + cff_dn * self.Kt_old[j,k_w-1]
            dAt_bl              = cff * ( self.Kt_old[j,k_w] - self.Kt_old[j,k_w-1])
            self.Gt1_bot[j]     = At_bl / (self.hbbl[j] * ws + self.eps)
            self.dGt1_dS_bot[j] = np.min([0,-dAt_bl/(ws+self.eps)])

            


    def compute_mixing_coefficients_surf(self):
        """ Compute boundary layer mixing coefficeints for surface KPP"""
        [Ly,N] = self.b.shape
        z_u_w  = self.grid_dict['z_u_w']

        # SET UP NEW MIXING COEFFICIENT ARRAYS
        self.Kv_surf = np.zeros([Ly,N+1])
        self.Kt_surf = np.zeros([Ly,N+1])
      
        self.ghat = np.zeros([Ly,N+1])
        

        #################################
        # 	SURFACE KPP
        ################################
        #---> j-loop
        
        self.wm2 = []
        self.ws2 = []
        self.sigma_y = []
        for j in range(Ly):
            #--> k-loop (top to kbl[j])
            # in fortran k=N-1,kbl(j),-1
            for k in range(N-1,self.kbl[j]-1,-1):
                k_w = k
                k_r = k-1

                Bfsfc = self.Bfsfc_bl[j]
                zscale = z_u_w[j,N] - z_u_w[j,k_w]
                
                # CALCULATE TURBULENT VELOCITY SCALES
                wm,ws = self.lmd_wscale_wm_and_ws(Bfsfc,zscale,self.ustar[j],self.hbls[j])
                self.wm2.append(wm)
                self.ws2.append(ws)
                # COMPUTE VERTICAL MIXING COEFFICIENTS
                sigma = (z_u_w[j,N] - z_u_w[j,k_w])  / np.max([self.hbls[j],self.eps])
                self.sigma1 = sigma #for debugging
                if j == 25:               
                   self.sigma_y.append(sigma)
                a1 = sigma - 2.
                a2 = 3.-2.*sigma
                a3 = sigma - 1.

                if sigma < 0.07:
                   cff = 0.5 * (sigma-0.07)**2/0.07
                else:
                   cff = 0
            
 
                if k == N-1: 
                   self.wm_debug = wm
                   self.hbls_debug = self.hbls[j]
                   self.cff_debug  = cff
                   self.sigma_debug = sigma
                   self.a1_debug    = a1
                   self.a2_debug    = a2
                   self.a3_debug    = a3

                self.Kv_surf[j,k_w] = wm * self.hbls[j] * ( cff + sigma * (1. + sigma * (\
                                                      a1 + a2*self.Gm1[j]+a3*self.dGm1_dS[j])))

                if k == N-1:
                   self.ws_debug = ws
                   self.hbls_debug = self.hbls[j]
                   self.cff_debug  = cff
                   self.sigma_debug = sigma
                   self.a1_debug    = a1
                   self.a2_debug    = a2
                   self.a3_debug    = a3
                
                self.Kt_surf[j,k_w] = ws * self.hbls[j] * ( cff + sigma * (1. + sigma * (\
                                                      a1 + a2*self.Gt1[j]+a3*self.dGt1_dS[j])))
            #---> end k-loop 
                if self.LMD_NONLOCAL:
                   if Bfsfc < 0:
                      self.ghat[j,k_w] = 0
                      self.ghat[j,k_w] = self.Cg * sigma * (1.-sigma)**2
                   else:
                       self.ghat[j,k_w] = 0.

            # ADD CONVECTIVE ADJUSTMENT IN SURFACE MIXED LAYER               
            if self.LMD_CONVEC and self.MLCONVEC: 
               for k in range(N-1,int(self.kbl[j]-1),-1):
                   k_w = k
                   k_r = k -1

                   if self.bvf[j,k_w] < 0:
                      self.Kt_surf[j,k_w] = self.Kt_surf[j,k_w] + self.ffac*self.nu0c

            # ADD CONVECTIVE ADJUSTMENT BELOW SURFACE MIXED LAYER
            # IF BKPP IS SWITCHED OFF!!
            for k in range(int(self.kbl[j]-1),-1,-1):
                k_w = k
                k_r = k -1
                if self.LMD_NONLOCAL:
                    self.ghat[j,k_w] = 0
                if self.LMD_CONVEC and self.LMD_BKPP == False:
                   if self.bvf[j,k_w] < 0:
                      self.Kv_surf[j,k_w] = self.Kv_surf[j,k_w] + self.nu0c
                      self.Kt_surf[j,k_w] = self.Kt_surf[j,k_w] + self.nu0c
                     

        #---> end j-loop




     


    def compute_mixing_coefficients_bot(self):
        """ Compute boundary layer mixing coefficients for bottom kpp """
        [Ly,N] = self.b.shape
        z_u_w  = self.grid_dict['z_u_w']

        v_upts = TTTW_func.v2u(self.v)

        self.sigma_bot = []
        self.Kv0 = np.zeros([Ly,N+1])
        self.Kt0 = np.zeros([Ly,N+1])
        for j in range(Ly):
            # turbulent velocity sclaes with buoyancy effects neglected
            ustar2 = self.r_D[j] * np.sqrt(self.u[j,0]**2 + v_upts[j,0]**2)
            wm = self.vonKar * np.sqrt(ustar2)
            ws = wm
 
            for k in range(1,N):
               k_w = k
               k_r = k - 1

               if k_w < self.kbl[j]: # NEED Zob
                  sigma = np.min(  [ ((z_u_w[j,k_w] - z_u_w[j,0] + self.Zob) / (self.hbbl[j] + self.Zob)),1.])
                  if j ==1:
                     self.sigma_bot.append(sigma)
                  a1 = sigma - 2.
                  a2 = 3. - 2.*sigma
                  a3 = sigma - 1.

                  self.Kv0[j,k_w] = wm * self.hbbl[j] * ( sigma * (1. + sigma * ( a1 + a2*self.Gm1_bot[j]+a3*self.dGm1_dS_bot[j]))) 
                  self.Kt0[j,k_w] = ws * self.hbbl[j] * ( sigma * (1. + sigma * ( a1 + a2*self.Gt1_bot[j]+a3*self.dGt1_dS_bot[j]))) 
                  

    def finalize_Kv_Kt(self):
        """ Finalize mixing coefficients """
        [Ly,N] = self.b.shape
        z_u_w  = self.grid_dict['z_u_w']        


        # ARRAYS SENT BACK TO TTTW SYSTEM FOR USE IN TIME STEPPING AND
        # CALCULATING OF FLOW FIELDS
        self.Kv_out = np.zeros([Ly,N+1])
        self.Kt_out = np.zeros([Ly,N+1])

        self.Kv_full = np.copy(self.Kv_surf)
        self.Kt_full = np.copy(self.Kt_surf)
        # IF BOTTOM KPP, CHECK FOR OVERLAP AND TAKE MAX MIXING COEFFICIENT
        if self.LMD_BKPP:
           self.Kv_temp = np.copy(self.Kv0)
           self.Kt_temp = np.copy(self.Kt0)
           for j in range(Ly):
                for k in range(1,N):
                    k_w = k
                    k_r = k -1

                    if k_w < self.kbl[j]:
                        # IF BBL REACHES INTO SBL, TAKE MAX OF SURFACE AND BOTTOM VALUES
                        z_bl = z_u_w[j,N]  - self.hbls[j]
                        if z_u_w[j,k_w] > z_bl:
                           self.Kv_temp[j,k_w] = np.max([self.Kv_surf[j,k_w],self.Kv0[j,k_w]])
                           self.Kt_temp[j,k_w] = np.max([self.Kt_surf[j,k_w],self.Kt0[j,k_w]])
                
                        self.Kv_full[j,k_w] = self.Kv_temp[j,k_w]
                        self.Kt_full[j,k_w] = self.Kt_temp[j,k_w]
                      
                      
                        #INCREASE DIFFUSIVITY IF CONVECTION ENCOUNTERED (bvf negative)
                        if self.BBL_CONVEC:
                           if self.bvf[j,k_w] < 0:
                              self.Kt_full[j,k_w] = self.Kt_full[j,k_w] + self.ffac*self.nu0c
                        


                    # ADD CONVECTIVE ADJUSTMENT OUTSIDE OF MIXED LAYERS
                    else:    
                        if self.LMD_CONVEC:
                           if self.bvf[j,k_w] < 0:
                              if self.LMD_KPP:
                                 z_bl = z_u_w[j,N] - self.hbls[j]
                                 if z_u_w[j,k_w] < z_bl:
                                    self.Kv_full[j,k_w] = self.Kv_full[j,k_w] + self.nu0c
                                    self.Kt_full[j,k_w] = self.Kt_full[j,k_w] + self.nu0c



        ### NOW COMPLETELY FINALIZE ###
        for j in range(Ly):
            for k in range(N+1):
                self.Kv_out[j,k] = np.max([self.Kv_full[j,k],self.my_Kv_bak])
                self.Kt_out[j,k] = np.max([self.Kt_full[j,k],self.my_Kt_bak])







