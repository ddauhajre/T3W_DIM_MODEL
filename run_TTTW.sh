#!/bin/sh
###################################################
# RUN T3W WITH KPP HOOKED UP
# CALL PYTHON SCRIPTS TO RUN ALL CODE
# AND SAVE FILES INTO NPY (ICS) and NETCDF (OUTPUT)
##################################################


echo " FOLLOWING PROCESSES WILL BE PERFORMED: "
echo " 		1) Calculate initial conditions with KPP through iterative process"
echo " 		2) Time stepping of TTTW system 				 " 

#############################
# CALCULATE INITIAL CONDITONS
#############################
make_IC=true 
#make_IC=false
if $make_IC ; then
   python src_code/TTTW_set_IC.py
fi
echo "				 "
echo " 				"
echo "			INITIAL CONDITONS CALCULATED"
echo "			"
echo "			"

###########################
# RUN TTTW TIME STEPPING
###########################
python src_code/TTTW_main.py

echo " 			TIME STEPPING DONE "

