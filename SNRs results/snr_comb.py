# cosmology related modules
import astropy
from astropy.cosmology import Planck18
from astropy.constants import c
from astropy import units as u
# for catalogues generation
from scipy.stats import uniform
import random as rnd
from scipy import stats
#from sklearn.neighbors import KernelDensity

#general tools
import sys
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.fftpack
from numpy.fft import *
from scipy.integrate import quad
import scipy.special
#from statistics import mean
from matplotlib import pyplot as plt 
import matplotlib.colors as colors
import math

#superrad
import os
from pathlib import Path
current = os.path.dirname(os.path.realpath('__file__'))
parent = os.path.dirname(current)
sys.path.append(parent)
from superrad.ultralight_boson import UltralightBoson
from superrad.cloud_model import CloudModel
# cosmology related modules
import astropy
from astropy.cosmology import Planck18
from astropy.constants import c
from astropy import units as u


"""
Columns:

* 1st: BH mas
* 2nd: BH spin
* 3d: redshift
* 4th: mopt
* 5th: weights= Wps (DL/(z+1))^2

"""

# General useful quantities:
sec_hour = 3600.0 #sec in an hour
sec_year = 3.154*10**7 #sec in a year
sec_week= 604800 # in sec



# Use cosmology consistent with Planck data 
cosmo = Planck18
cu = c.to('Mpc/yr').value # speed of light in Mpc per year
H0 = Planck18.H0.value # km/s/Mpc
cuk = c.to('km/s').value #km/s

# LISA / mission/ astro/ cosmo
tmiss = 4*sec_year #LISA mission duration 
#tint = np.linspace(0,tmiss,17) # 4 years in 16 intervals
#dtmiss=(tint[1]-tint[0])/sec_year # in yrs




def main():
    """
    t=np.linspace(0,17709507,21,dtype=int)
    q3ndsnrs=[]
    for i in range(len(t)-1):
        q3nds=np.loadtxt("snrtimes_{}_{}.dat".format(t[i], t[i+1]))
        q3ndsnrs.append(q3nds)
        print("snrtimes_{}_{}.dat".format(t[i], t[i+1]),"loaded")
    
    q3ndsnrs=np.array(q3ndsnrs)
    file_name = "q3ndsnrst.dat"
    # Save the array to a .dat file
    np.savetxt(file_name, q3ndsnrs, delimiter=' ')
    print(file_name, "saved")
    """
    
    snrst=np.loadtxt("q3ndsnrst.dat")
    data=np.loadtxt("q3nod10.dat")
    datasnr=np.column_stack((data, snrst))
    
    file_name = "data_snrs.dat"
    # Save the array to a .dat file
    np.savetxt(file_name, datasnr, delimiter=' ')
    print(file_name, "saved")

    
main()
    
    