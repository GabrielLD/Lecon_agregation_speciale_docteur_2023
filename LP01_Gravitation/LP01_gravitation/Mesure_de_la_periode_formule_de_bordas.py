#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:13:47 2023

@author: gld
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo


#%% Formule de Borda Mesures

# pour 5 degrés: 
T = [1.2499, 1.2526, 1.2504, 1.2517, 1.2520, 1.2519]
Tmean = np.mean(T)
print(Tmean)
Tstd = np.std(T)
print(Tstd)
uT1 =2*Tstd/np.sqrt(len(T))

L= 39e-2 # m longueur du fil
uL = 5e-3/(2*np.sqrt(3)) # m


# Mesures de la période en fonction de l'angle de départ
theta = [5, 10,15,20,25,30,35, 40, 45, 55, 60, 70,80]# radians

thetarad = [i*np.pi/180 for i in theta]

uTheta = 2/(2*np.sqrt(3))
uthetarad = uTheta*np.pi/180

T = [1.247, 1.2513,1.2535,1.2574,1.2620,1.2704,  1.2789,1.286,1.2954, 1.3191,1.3362, 1.3726,1.418] # s
uT = [i*0.01 for i in T]

plt.figure()
plt.errorbar(thetarad, T, xerr=uthetarad, yerr =uT, fmt='+')


#%% incertitudes en suivant le Diffon

# Données expérimentales
x = np.array(thetarad) # angle de départ du pendule en radians
y = np.array(T) # période mesurée
ux = np.array(uthetarad) # incertitudes pour l'angle
uy = np.array(uT) # incertitudes pour la periode


def f(x,p):
    "Formule de Borda "
    T0 = p
    #T0, a = p
    return T0*(1+((x)**2)/16 +(11/3072)*(x)**4)
    #return  T0*(1+((x+a)**2)/16 +(11/3072)*(x+a)**4)

# derivee de la fonction f par rapport à la variable de controle x
def Dx_f(x,p):
    T#0, a = p
    T0 =p
    return T0*(1/16*(2*x)+11/3072*(4*(x)**3))

#fonction d'écart pondérée par les erreurs
def residual(p, y, x):
    return (y-f(x,p))/np.sqrt(uy**2+(Dx_f(x,p)*ux)**2)

# estimation initiale des paramètres
# elle ne joue aucun rôle (initialise le tableau des valeurs)
# permet d'aider l'ajustement si compliqué
p0 = np.array([0])
# on utilise l'algorithme des moindes carrés non-linéaires disponible dans scipy

result = spo.leastsq(residual, p0, args = (y,x), full_output=True)

# on obtient les paramètres d'ajustement optimaux
# correspond à T0 dans la formule qui est le seul paramètre d'ajustement
popt = result[0] 
pcov =result[1]
# incertitudes types 
upopt = np.sqrt(np.abs(np.diagonal(pcov)))
# on détermine le chi2
chi2r = np.sum(np.square(residual(popt,y,x)))/(x.size-popt.size)
print(chi2r)

#%% Tracer de la courbe experimentale et modèle ajusté
abscisse = np.linspace(0, 1.5, 100)
plt.figure()
plt.errorbar(thetarad, T, xerr=uthetarad, yerr =uT, fmt='+', label = 'Mesures expérimentales')
plt.plot(abscisse,f(abscisse, popt), label = r'Modèle, formule de Borda')
plt.xlabel(r'$\theta~(^\circ)$')
plt.ylabel(r'$T (s)$')
plt.legend() 
T0 = popt[0]
uT0 = upopt[0]

print('T0 = ' + str(np.round(T0, 3)) + '+/- ' + str(np.round(uT0,3))+ 's')

#%% Mesure de g
# t0 = 2*np.pi*sqrt(L/g)
g = (2*np.pi/T0)**2*L
ug = g*np.sqrt(2*(uL/L)**2+2*(uT0/T0)**2)

print('On mesure g ='+ str(np.round(g,3)) + '+/-' + str(np.round(ug,4)) +'m²/s')
print('À Paris g = 9.812 m²/s')