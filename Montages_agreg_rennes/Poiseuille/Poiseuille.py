#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:35:24 2023

@author: gld
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

#%%

lprime= np.array([7, 8.9, 11.3, 13.4, 16.16, 18.4]) # cm distance dépasse du vase de Mariotte
t = np.array([61, 49, 62, 58, 57, 64]) # s
m = np.array([4.01, 4.2389, 7.187, 8.78, 10.0589, 12.579]) # g
um = 0.01*np.ones(len(m)) 
#%% Mesures 
Dm = m/t
uDm = Dm*np.sqrt((um/m)**2)

Lma = 31 # cm
Ltot = 33.1 # cm 
h = Lma - (Ltot-lprime)
#%% Rayon du tuyau
g = 9.81 # m²/s
meau = np.array([1.908, 2.555, 2.243]) # g
meau_mean = np.mean(meau) 
meau_std = np.std(meau)
umeau = 2*meau_std/np.sqrt(len(meau))
Ltuyau = 1.185 # m
uL = (1.185-1.180)/2/np.sqrt(3)
rho = 1000 # kg/m^3
Rtuyau = np.sqrt(meau_mean*1e-3/(rho*np.pi*Ltuyau)) # kg
print("Le rayon du tuyau est de "+ str(np.round(Rtuyau, 6)) + " m")

#%%

x = h*1e-2
ux = x*np.sqrt((1e-3/Lma/1e-2)**2+(1e-3/ltot/1e-2)**2+(1e-3/lprime/1e-2)**2)


y = Dm*1e-3
uy = uDm*1e-3


def f(x,p):
    "Pertes de charges"
    eta = p
    return x*rho**2*g*np.pi*(Rtuyau)**4/(8*eta*Ltuyau)

# derivee de la fonction f par rapport à la variable de controle x
def Dx_f(x,p):
    eta =p
    return rho**2*g*np.pi*(Rtuyau)**4/(8*eta*Ltuyau)
#fonction d'écart pondérée par les erreurs
def residual(p, y, x):
    return (y-f(x,p))/np.sqrt(uy**2+(Dx_f(x,p)*ux)**2)

p0 = np.array([1e-3])
result = spo.leastsq(residual, p0, args = (y,x), full_output=True)

popt = result[0] 
pcov =result[1]
# incertitudes types 
upopt = np.sqrt(np.abs(np.diagonal(pcov)))
# on détermine le chi2
chi2r = np.sum(np.square(residual(popt,y,x)))/(x.size-popt.size)
print(chi2r)

#%%
plt.figure()
plt.errorbar(x, y, xerr=ux, yerr=uy, fmt =  '+')
abscisse = np.linspace(0,np.max(x),100)
plt.plot(abscisse, f(abscisse, popt))
plt.xlabel(r'h (m)')
plt.ylabel(r'Dm (kg/s)')
