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
# from SuppeRad
bc = UltralightBoson(spin=1, model="relativistic")


q3nd = np.loadtxt("extq3nd.dat") # the data from the catalogs




# General useful quantities:
sec_hour = 3600.0 #sec in an hour
sec_year = 3.154*10**7 #sec in a year
# Use cosmology consistent with Planck data 
cosmo = Planck18
cu = c.to('Mpc/yr').value # speed of light in Mpc per year
H0 = Planck18.H0.value # km/s/Mpc
cuk = c.to('km/s').value #km/s

# LISA / mission/ astro/ cosmo
tmiss = 4*sec_year #LISA mission duration
tint = np.linspace(0,tmiss,17) # 4 years in 16 intervals
dtmiss=(tint[1]-tint[0])/sec_year # in yrs

# Poission distribution
def poissond(k,y,t):
    # x = events occuring 
    # y = dN/dt
    # t = time range of interest
    
    l = y*t
    return np.exp(-l)*l**k/np.math.factorial(k)

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

def snr(M,m,a,theta,zz,tm):
    
    
    dObs =Planck18.luminosity_distance(zz).value # Mpc
    
    wf = bc.make_waveform(M, a, m, units="physical")
    
    tc=wf.cloud_growth_time()/sec_hour;  # Cloud growth time in hours in src frame
    tgw = wf.gw_time()/sec_hour*(1+zz); # timescale of GW emission in hours in det frame
    tsrc0 = 0
    tsrcf = tm/(1+zz) - tc*sec_hour # in source frame
    
    h0p,h0x,delta = wf.strain_amp(tsrc0, theta, dObs) # Strain 
    fgw0 = wf.freq_gw(tsrc0)/(1+zz) #min frequency in detector frame 
    fmin=fgw0
    fmax=wf.freq_gw(tsrcf)/(1+zz) #max frequency in detector frame 
    #phi = wf.phase_gw(tsrc)
    
    
    
    f0 = wf.freq_gw(10**8*tgw)/(1+zz) #Hz
    δf=fgw0-f0 #Hz
    fex=np.linspace(fmin,fmax,10**4)
                                 
    int1=(np.abs(Hf1(fex,tgw,f0,δf,h0p))**2 + np.abs(Hf1(fex,tgw,f0,δf,h0x))**2)/S_n(fex)
    integral = np.trapz(int1, fex)
    snr = 2*np.sqrt(integral)
    
    return snr 


# order of magnitude
def ord_magn(x):
    return math.floor(math.log10(abs(x)))



# To make realizations: (b,e) defines the beginning and ending number of them. CAUTION e is not included, so set desired+1.
def realizations(b,e):
    q3ndrows = q3nd.shape[0]
    q3ndsnr1=[]
    for i in range(b,e):
        for j in range(16):
            for row in range(q3ndrows):
                wt = q3nd[row,-1] # in yrs^-1
                p1 = poissond(1,wt,dtmiss)
                #p2=poissond(2,wt,dtmiss)
                #p3=poissond(3,wt,dtmiss)
                #p4=poissond(4,wt,dtmiss)
                cv = np.random.uniform(0,1) # random value for comparison

                if cv <= p1:
                    Mn = q3nd[row,0]
                    an = q3nd[row,1]
                    zn = q3nd[row,2]
                    moptn = q3nd[row,-2]

                    # draw angle
                    rand=np.random.uniform(-1,1)
                    thn = np.arcsin(rand)+np.pi/2
                    # time of merger is (j+1)*dtmiss after the beginning of the mission
                    tmn = tmiss-(j+1)*dtmiss*sec_year

                    nsnropt = snr(Mn,moptn,an,thn,zn,tmn)

                    if nsnropt<0.1 or math.isnan(nsnropt):
                        continue

                    if nsnropt>=0.1 and nsnropt<1:
                        mu = moptn+0.025*10**ord_magn(moptn)
                        md = moptn-0.025*10**ord_magn(moptn)

                        nsnru = snr(Mn,mu,an,thn,zn,tmn)
                        nsnrd = snr(Mn,md,an,thn,zn,tmn)

                        while nsnru>nsnropt:
                            if nsnru>=1:
                                listn=[Mn,an,zn,thn,tmn,moptn,wt,mu,nsnru,i]
                                q3ndsnr1.append(listn)
                                #print(i,j,row)
                            mu = mu+0.025*10**ord_magn(moptn)
                            nsnru = snr(Mn,mu,an,thn,zn,tmn)

                        while nsnrd>nsnropt:
                            if nsnrd>=1:
                                listn=[Mn,an,zn,thn,tmn,moptn,wt,md,nsnrd,i]
                                q3ndsnr1.append(listn)
                                #print(i,j,row)
                            md = md-0.025*10**ord_magn(moptn)
                            nsnrd = snr(Mn,md,an,thn,zn,tmn)

                    if nsnropt>=1:

                        listropt=[Mn,an,zn,thn,tmn,moptn,wt,moptn,nsnropt,i]
                        q3ndsnr1.append(listropt)

                        mu = moptn+0.025*10**ord_magn(moptn)
                        md = moptn-0.025*10**ord_magn(moptn)

                        nsnru = snr(Mn,mu,an,thn,zn,tmn)
                        nsnrd = snr(Mn,md,an,thn,zn,tmn)
                        while nsnru>=1:
                            listn=[Mn,an,zn,thn,tmn,moptn,wt,mu,nsnru,i]
                            q3ndsnr1.append(listn)
                            #print(i,j,row)
                            mu = mu+0.025*10**ord_magn(moptn)
                            nsnru = snr(Mn,mu,an,thn,zn,tmn)

                        while nsnrd>=1:
                            listn=[Mn,an,zn,thn,tmn,moptn,wt,md,nsnrd,i]
                            q3ndsnr1.append(listn)
                            #print(i,j,row)
                            md = md-0.025*10**ord_magn(moptn)
                            nsnrd = snr(Mn,md,an,thn,zn,tmn)

        print(i)
    return q3ndsnr1

def main():
    s = sys.argv[1:]
    initial = int(s[0])
    final = int(s[1])
    
    
    q3nd1=realizations(initial,final+1) 
    q3nd1ar=np.array(q3nd1) 
    file_name = "q3nd1_{}_{}.dat".format(initial, final)
    # Save the array to a .dat file
    np.savetxt(file_name, q3nd1ar, delimiter=' ')
    print(file_name, "saved")
    
    
main()
