#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:18:54 2023

@author: gld
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo


T = np.array([81.6, 78.8, 75.8, 73.2, 70.5, 67.1, 65.2, 55.6, 48.9]) # degres
uT = np.ones(len(T))*0.2#np.array([0.5,0.5,0.5,0.2, 0.2, 0.2])#np.ones(len(T))*0.5


#30 mesures
Rmean = np.array([1.656, 1.644, 1.63, 1.61, 1.597, 1.580, 1.5729,1.524, 1.491]) # ohm
Rstd = np.array([304.41,434.9, 405.38, 285.74, 641.04, 656.75, 368.68, 224.66, 708.90])*1e-6 # ohm
uR = 1.8*Rstd/np.sqrt(len(Rmean))



plt.figure()
plt.errorbar(T, Rmean, xerr = uT, yerr = uR, fmt='+')
plt.xlabel('T')
plt.ylabel('Ohm')

#%%
T0 = 273.15 # 

x = T+273-T0
ux = uT
y = Rmean
uy = uR

def f(x,p):
    "Résistance en fonction de la température"
    a, R0 = p
    return a*x+R0


# derivee de la fonction f par rapport à la variable de controle x
def Dx_f(x,p):
    a, R0 = p
    return a
#fonction d'écart pondérée par les erreurs
def residual(p, y, x):
    return (y-f(x,p))/np.sqrt(uy**2+(Dx_f(x,p)*ux)**2)

p0 = np.array([0,0])
result = spo.leastsq(residual, p0, args = (y,x), full_output=True)

popt = result[0] 
pcov =result[1]
# incertitudes types 
upopt = np.sqrt(np.abs(np.diagonal(pcov)))
# on détermine le chi2
chi2r = np.sum(np.square(residual(popt,y,x)))/(x.size-popt.size)
print(chi2r)

plt.figure()
abscisse = np.linspace(np.min(T)-1, np.max(T), 100)
plt.plot(abscisse, f(abscisse, popt))
plt.errorbar(T, Rmean, xerr = uT, yerr = uR, fmt='+')
plt.xlabel('T')
plt.ylabel('Ohm')
a = popt[0]
print(a)
#%%
R = 0.5e-3/2 # m rayon du fil
L = 15 # longueur du fil
alphatheo = 67.6e-12 # Ohm m / K

alpha = a*np.pi*R**2/L
print(alpha)