#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:59:32 2021

@author: aoife
"""
import numpy as np
import math
import matplotlib.pyplot as plt 
from scipy  import optimize
from numpy import *
import statistics as stat
import glob 
from scipy.stats import chisquare

from PIL import Image
from PIL import ImageFilter
import timeit

f_obs=[0.9,2.1,2.9,4.1]
f_exp=[1,2,3,4]


chi,p=chisquare(f_obs=f_obs,f_exp=f_exp)

print(chi,p)



data = np.loadtxt('meteorology.csv', delimiter=',', skiprows=1)
time = data[:,0]
Rn = data[:,1]  # net radiation (W/m2)
G  = data[:,2]  # ground heat flux (W/m2)
H  = data[:,3]  # sensible heat flux (W/m2)
LE = data[:,4]  # latent heat flux (W/m2)
q  = data[:,5]  # specific humidity (kg/kg)
T  = data[:,6]  # air temperature (C)
Ts = data[:,7]  # surface temperature (C)
U  = data[:,8]  # wind speed (m/s)


plt.plot(time, G,'r', Label="Ground Heat Flux") 
plt.plot(time, H,'b',Label="Sensible Heat Flux") 
plt.plot(time, LE,'g',Label="Latent Heat Flux")
plt.plot(time, Rn,Label="net radiation")

plt.xlabel("Time (hours)")
plt.ylabel("Net Radiation / heat Flux (W/m^2)")
plt.title("Net Radiation and Flux Variations across a Day")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

