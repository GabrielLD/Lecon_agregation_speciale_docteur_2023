#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 23:01:04 2023

@author: gld

Tentative de traiter verifier le modèle pour le temps de vie de la 
radioactivite alpha
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression 
import scipy.optimize as spo
import matplotlib as mpl

plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"

params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          }
plt.rcParams.update(params)
#%%
df = pd.read_csv('Radioactivitealpha_thibault.csv')
#df = pd.read_csv('Radioactivitealpha.csv')
#%%
energie = df['E (MeV)'].to_numpy()  # MeV
tempsdemivie = df['tauhalf'].to_numpy()
Z = df['Z'].to_numpy()
A = df['A'].to_numpy()
uy = 0.9746*np.ones(len(Z))
ux = np.ones(len(Z))
lntempsdemivie_exp = np.log(tempsdemivie)
plt.figure()
plt.plot(1/np.sqrt(energie), lntempsdemivie_exp, 'o')
#%%
# Modelisation à partir du calcul du temps de demi vie
e = 1.602E-19 #C
hbar = 1.055E-34 # 
malpha = 6.646E-27 # g
r0 = 1.4e-15#fm
epsilon0 = 8.854E-12 #
a=4*e/hbar*np.sqrt(malpha*(Z-2)*r0/(np.pi*epsilon0))*pow(A,1/6)
b=e**2*(Z-2)/(2*hbar*epsilon0)*np.sqrt(2*malpha)
print(b)

lntn=np.log(2*r0*np.sqrt(malpha/(2*energie))*pow(A,1/3))
lnT = a-b/np.sqrt(energie*e*1e6) # coefficient de transmission
lnth=lntn+np.log(np.log(2))-lnT
invsqrtE=1/np.sqrt(energie)

plt.figure()
plt.errorbar(1/np.sqrt(energie), lntempsdemivie_exp, yerr=sigma_y, fmt='o', label = 'exp')
plt.plot(1/np.sqrt(energie), lnth, '^', label = 'modèle')
plt.legend()

#%% Tentative de calcul de chi2r en suivant le Diffon

# scipy ajustement 
def affine(x, a, b): 
    return a*x + b

def f(x,p):
    a,b =p
    return a*x+b
    
def Dx_f(x,p):
    a,b = p
    return a

def residual(p,y,x):
    return (y-f(x,p))/np.sqrt(uy**2+(Dx_f(x,p)*ux)**2)
p0=np.array([0,0])

result=spo.leastsq(residual, p0, args=(1/np.sqrt(energie), lntempsdemivie_exp), full_output=True)

#On obtient les params d'ajustement
popt=result[0]
# la matrice de variance-covariance
pcov = result[1]
# les incertitudes types
upopt=np.sqrt(np.abs(np.diagonal(pcov)))

chi2r = np.sum(np.square(residual(popt,lntempsdemivie_exp,1/np.sqrt(energie))))/(lntempsdemivie_exp.size-popt.size)
print(chi2r)
#%% Tracé des courbes avec ajustement affine
params, covs = curve_fit(affine, 1/np.sqrt(energie), lntempsdemivie_exp)
print(params)
a_exp= np.round(params[0],8)
b_exp= np.round(params[1],8)
abscisse = np.linspace(0.3,0.7,100)
#labelfit = r"$\gamma$ = " + str(a) + "N/m"
fitexp=affine(abscisse,a_exp,b_exp)
print(a)
print(b)
params, covs = curve_fit(affine, 1/np.sqrt(energie), lnth)
print(params)
atheo = np.round(params[0],8)
btheo= np.round(params[1],8)


plt.figure()
plt.plot(1/np.sqrt(energie), lntempsdemivie_exp,'o', label =r'Mesures exp')
plt.plot(abscisse,fitexp,color='k', label = r'$f(x) = 328.5x-121.5$')
plt.plot(1/np.sqrt(energie), lnth, '^', label = 'modèle')
plt.plot(abscisse,affine(abscisse,atheo,btheo),'--',color='k', label = r'$f(x) = 360.4x-148.9$' )
plt.axis([0.35, 0.55, -30, 50])
plt.xlabel(r'$E^{-1/2}$ (Mev)$^{-1/2}$')
plt.ylabel(r'$\ln(\tau_{1/2})$')
plt.legend()
plt.savefig('Radioactivite_alpha.png', dpi=300)
plt.show()



#%% Correlation modele _ vs _ exp
plt.figure()
plt.plot(lntempsdemivie_exp,lnth, 'o')
params, covs = curve_fit(affine, lntempsdemivie_exp, lnth)
print(params)
a = np.round(params[0],8)
b= np.round(params[1],8)
abscisse2 = np.linspace(0,43,100)
plt.plot(abscisse2, affine(abscisse2, a,b),color='k', label=r'$f(x) = 1.08x-15.2$')
plt.xlabel(r'$\ln(\tau_{1/2})$ exp')
plt.ylabel(r'$\ln(\tau_{1/2})$ theo')
plt.legend()
plt.savefig('Modele_vs_exp.png', dpi= 300)