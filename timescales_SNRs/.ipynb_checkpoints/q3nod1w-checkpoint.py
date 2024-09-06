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



# from SuppeRad
bc = UltralightBoson(spin=1, model="relativistic")






# SNR

# Sensitivity curve
fstar=19.09*10**(-3) #Hz
L=2.5*10**9 #in m
A=9*10**(-45)
alpha = 0.166
beta=299
kappa=611
gamma=1340
fk=0.00173


def Poms(f):
    return ((1.5*10**(-11))**2)*(1+((2*10**(-3))/f)**4)

def Pacc(f):
    return ((3*10**(-15))**2)*(1+((0.4*10**(-3))/f)**2)*(1+(f/(8*10**(-3)))**4)

def Pn(f):
    return (Poms(f)/(L**2))+2*(1+(np.cos(f/fstar)**2))*Pacc(f)/(((2*np.pi*f)**4)*L**2)

def R(f):
    return 0.3/(1+0.6*(f/fstar)**2)

def Sc(f):
    return A*f**(-7/3)*np.exp(f**alpha+beta*f*np.sin(kappa*f))*(1+np.tanh(gamma*(fk-f)))

def S_n(f):
    return Pn(f)/R(f) + Sc(f)

# SNR calculation
def t01(f,tg,finf,Df):
    return tg*(Df/(f-finf)-1)

def ϕ01(f,tg,finf,Df):
    return (finf-f)*t01(f,tg,finf,Df)*sec_hour+Df*tg*sec_hour*np.log(Df/(f-finf))

def ddϕ01(f,tg,finf,Df):
    return -((f-finf)**2)/(Df*tg*sec_hour)

def Hf1(f,tg,finf,Df,h):
    return h/(Df/(f-finf))*np.exp(1j*2*np.pi*ϕ01(f,tg,finf,Df))*np.exp(1j*np.pi/4)*np.sqrt(1/(4*ddϕ01(f,tg,finf,Df)))
# it is already devided by 2

# I am exporting the tgw as well here!

def snr(M,m,a,theta,zz,tm,times,i,timescalesarray):
    
    
    dObs =Planck18.luminosity_distance(zz).value # Mpc
    
    wf = bc.make_waveform(M, a, m, units="physical")
    
    tc=wf.cloud_growth_time()/sec_hour;  # Cloud growth time in hours in src frame
    tgw = wf.gw_time()/sec_hour*(1+zz); # timescale of GW emission in hours in det frame
    timescalesarray.append([tgw,tc*(1+zz)])
    tsrc0 = 0
    fgw0 = wf.freq_gw(tsrc0)/(1+zz) #min frequency in detector frame
    f0 = wf.freq_gw(10**8*tgw)/(1+zz) #Hz
    δf=fgw0-f0 #Hz
    h0p,h0x,delta = wf.strain_amp(tsrc0, theta, dObs) # Strain 
    fmin=fgw0
    snrs=[]
    for j in range(len(times)):
        td = times[j]
        tsrcf = min(tm/(1+zz) - tc*sec_hour,td/(1+zz)) # in source frame , tm and td in seconds
        fmax=wf.freq_gw(tsrcf)/(1+zz) #max frequency in detector frame 
        #phi = wf.phase_gw(tsrc)
        fex=np.linspace(fmin,fmax,10**4)

        int1=(np.abs(Hf1(fex,tgw,f0,δf,h0p))**2 + np.abs(Hf1(fex,tgw,f0,δf,h0x))**2)/S_n(fex)
        integral = np.trapz(int1, fex)
        sn = 2*np.sqrt(integral)
        snrs.append(sn)
    
    return snrs # in hours in det frame


def main():
    
    inp = sys.argv[1:]
    initial = int(inp[0])
    final = int(inp[1])
  
    data=np.loadtxt("q3nod10.dat") #data with SNR>=10

    # Mn,an,zn,thn,tmn,moptn,wt,md,nsnrd,i

    print("data loaded")
    
    tt=np.array([sec_week,4*sec_week,26*sec_week, 52*sec_week])
    
    snrtimes=[]
    timescales=[]
    
    for i in range(initial,final):
        s=snr(data[i,0],data[i,-3],data[i,1],data[i,3],data[i,2],data[i,4],tt,i,timescales)
        snrtimes.append(s)
        
    snrtimes=np.array(snrtimes)
    timescales=np.array(timescales)
    
    
    file_name1 = "snrtimes_{}_{}.dat".format(initial, final-1)
    # Save the array to a .dat file
    np.savetxt(file_name1, snrtimes, delimiter=' ')
    print(file_name1, "saved")

    file_name2 = "timescales_{}_{}.dat".format(initial, final-1)
    # Save the array to a .dat file
    np.savetxt(file_name2, timescales, delimiter=' ')
    print(file_name1, "saved")
    
main()
