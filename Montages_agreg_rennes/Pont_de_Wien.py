#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:48:32 2023

@author: gld

Étude du Pont de Wien
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# Étude du filtre 
plt.clf()
plt.cla()
plt.close('all')

f = np.array([10   , 20    , 50    , 100  , 200   , 300  , 400   , 600   , 800  , 1000   ,1200 , 1600  , 3200 , 6400   , 12800  ,20850]) # Hz
E = np.array([1.71 ,1.71   , 1.71   , 1.73 , 1.73  ,1.72  , 1.72  , 1.71  ,1.71  , 1.72  ,1.72 ,  1.72  , 1.72 , 1.72   , 1.72   ,1.72]) # V
S = np.array([11e-3,22.6e-3, 51.1e-3, 89e-3, 124e-3,135e-3, 139e-3, 139e-3,135e-3, 130e-3,124e-3, 111e-3, 72e-3, 39.7e-3, 20.5e-3, 12.8e-3]) # V
phi = np.array([82 ,78.1   ,66    , 50.6 , 27.7  , 15   , 6.0   , -5.9  ,-15   , -22     ,-27.5, -37   , -59.1, -73    , -81.3  , -87.4]) # deg
B = S/E



def Bfun(f, B0, f0, Q):
    #B0, f0, Q = p
    return B0/np.abs(1+1j*(f/f0-f0/f)*Q)


abscisse = np.arange(0, 30e3)

popt, pcov = curve_fit(Bfun, f, B)

B0 = popt[0]
f0 = popt[1]
Q = popt[2]


#Q = 1/3
R = 10e3
C = 10e-9
#B0 = Q
#f0 = 1/(2*np.pi*R*C)
#p = [B0, f0, Q ]

plt.figure(1)
plt.plot(f, B, 'o')
#plt.plot(abscisse, Bfun(abscisse, p))
plt.plot(abscisse, Bfun(abscisse, *popt))
plt.xscale('log')
plt.xlabel(r'fréquence (Hz)')
plt.ylabel(r'B')


def phifun(f, f0, Q):
    return np.arctan(Q*(f/f0-f0/f))*180/np.pi

#popt, pcov = curve_fit(phifun, f, phi)


plt.figure(3)
plt.plot(f,phi, 'o')
plt.plot(abscisse, phifun(abscisse, f0, Q))
plt.xlabel(r'fréquence (Hz)')
plt.ylabel(r'phi')
plt.xscale('log')

#%% Système bouclé