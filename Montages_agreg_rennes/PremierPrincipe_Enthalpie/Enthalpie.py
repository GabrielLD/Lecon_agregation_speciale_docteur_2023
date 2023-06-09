#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 18:13:50 2023

@author: gld
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

#%%
t = [0, 1165]
ut =(1/2/np.sqrt(3))*np.ones(len(t))
m = [147.4, 147.3]
um = (0.2)*np.ones(len(m))

temps = [9, 20, 40, 50, 70, 93, 115, 144, 162, 175, 184, 200]
utemps =(1/2/np.sqrt(3))*np.ones(len(temps))
masse = [144.2, 142.3, 140.6, 139.6, 138.1, 136.0, 134.2, 132.1, 130.6, 129.4, 128.6, 127.4]
umasse = (0.3)*np.ones(len(masse))

plt.figure()
plt.errorbar(t, m, xerr= ut, yerr= um, fmt='+', label = 'sans chauffe')
plt.errorbar(temps, masse, xerr=utemps, yerr=umasse, fmt='*', label = 'avec chauffe')
plt.xlabel('temps')
plt.ylabel('masse')


#%% Pente sans chauffe

# Données expérimentales
x = np.array(t) # angle de départ du pendule en radians
y = np.array(m) # période mesurée
ux = ut # incertitudes pour l'angle
uy = um # incertitudes pour la periode


def f(x,p):
    "FOnction linéaire"
    a,b =p
    return a*x+b

# derivee de la fonction f par rapport à la variable de controle x
def Dx_f(x,p):
    a, b = p
    return a

#fonction d'écart pondérée par les erreurs
def residual(p, y, x):
    return (y-f(x,p))/np.sqrt(uy**2+(Dx_f(x,p)*ux)**2)

# estimation initiale des paramètres
# elle ne joue aucun rôle (initialise le tableau des valeurs)|
# permet d'aider l'ajustement si compliqué
p0 = np.array([0, 0])
# on utilise l'algorithme des moindes carrés non-linéaires disponible dans scipy

result = spo.leastsq(residual, p0, args = (y,x), full_output=True)

# on obtient les paramètres d'ajustement optimaux
# correspond à T0 dans la formule qui est le seul paramètre d'ajustement
popt0 = result[0] 
pcov0 =result[1]
# incertitudes types 
upopt0 = np.sqrt(np.abs(np.diagonal(pcov0)))
# on détermine le chi2
chi2r = np.sum(np.square(residual(popt0,y,x)))/(x.size-popt0.size)
print(chi2r)

dmdt_0 = popt0[0]
print(dmdt_0)

#%% Pente avec chauffe

# Données expérimentales
x = np.array(temps) # angle de départ du pendule en radians
y = np.array(masse) # période mesurée
ux = utemps# incertitudes pour l'angle
uy = umasse # incertitudes pour la periode


def f(x,p):
    "Fonction linéaire"
    a,b =p
    return a*x+b

# derivee de la fonction f par rapport à la variable de controle x
def Dx_f(x,p):
    a, b = p
    return a

#fonction d'écart pondérée par les erreurs
def residual(p, y, x):
    return (y-f(x,p))/np.sqrt(uy**2+(Dx_f(x,p)*ux)**2)

# estimation initiale des paramètres
# elle ne joue aucun rôle (initialise le tableau des valeurs)
# permet d'aider l'ajustement si compliqué
p0 = np.array([0, 0])
# on utilise l'algorithme des moindes carrés non-linéaires disponible dans scipy

result = spo.leastsq(residual, p0, args = (y,x), full_output=True)

# on obtient les paramètres d'ajustement optimaux
# correspond à T0 dans la formule qui est le seul paramètre d'ajustement
poptc = result[0] 
pcovc =result[1]
# incertitudes types 
upoptc = np.sqrt(np.abs(np.diagonal(pcovc)))
# on détermine le chi2
chi2r = np.sum(np.square(residual(poptc,y,x)))/(x.size-poptc.size)
print(chi2r)

dmdt_c = poptc[0]
print(dmdt_c)

plt.figure()
abscisse = np.linspace(0,200)
#plt.errorbar(t, m, xerr= ut, yerr= um, fmt='+', label = 'sans chauffe')
#plt.plot(abscisse, f(abscisse,popt0))
plt.errorbar(temps, masse, xerr=utemps, yerr=umasse, fmt='+', label = 'avec chauffe')
plt.plot(abscisse, f(abscisse, poptc))
plt.xlabel('temps')
plt.ylabel('masse')
#%%
U = 28.5 # V
I = 0.61 # A
Lv = U*I/(np.abs(dmdt_c)-np.abs(dmdt_0))
print(Lv)
Lvtheo = 198.5
udmdt0 = upopt0[0]
udmdtc = upoptc[0]
uLv = Lv*np.sqrt((udmdtc/dmdt_c)**2+(udmdt0/dmdt_0)**2)
print(uLv)
