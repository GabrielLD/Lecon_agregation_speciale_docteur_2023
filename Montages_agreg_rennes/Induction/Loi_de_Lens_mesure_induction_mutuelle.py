#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 09:48:58 2023

@author: gld

Mesure de l'inductance mutuelle poly induction de Philippe

On prend une bobine très longue type solénoide infini dans lequel on met une 
bobine plus petite. Il faut être dans l'approx du solenoide infini

e = -dphi/dt = -NSdB/dt = -mu0NSndi/dt = -Mdi/dt
M=mu0NSn

On prend le signal triangulaire car on peut vérifier que didt est la dérivation
du signal d'entrée
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo


#%% CHAMP MAGNETIQUE DANS LA BOBINE AU TESLAMETRE
R = 100.5
i = 2.06
U = i*R
print(U)
d = np.array([0, 2, 4,6, 8, 10, 12, 14, 16, 18, 20 , 22, 24, 26])
Bx = np.array([1.27, 1.27, 1.27, 1.27, 1.27, 1.27, 1.26, 1.25, 1.22, 1.12, 0.86, 0.41, 0.16, 0.08]) # mT
Bz = np.array([0.05, 0.04, 0.05, 0.06, 0.04, 0.05, 0.05,0.04 , 0.05, 0.03, 0.02, 0.01, 0.00, 0.00]) # mT

plt.figure()
plt.plot(d,Bx, 'o')
plt.plot(d,Bz, 'o')
plt.xlabel(r'distance')
plt.ylabel(r'Champ Magnétique')

#%% MESURE DU COEFFICIENT D'INDUCTANCE MUTUELLE ENTRE DEUX BOBINES
# mesures true rms aux bornes de la résistance
R = np.array(100.6) # Ohm
uR = R*0.8+3*0.1
Vpp = 5.0 # V
f =    np.array([400  ,600  ,800  ,1000 ,1200 ,1500]) # Hz
T = 1/f
Xeff = np.array([0.967,0.970,0.969,0.968,0.968,0.967]) # V
uXeff = Xeff*0.6%+5*0.001


Yeff = np.array([2.9  ,4.2  ,5.5  ,6.9  ,8.2  ,10.2]) # mV
Yeff = Yeff*1e-3 # V
uYeff = Yeff*0.6%+5*0.1
didt = 4*f*Xeff*np.sqrt(3)/R


#%%
x = didt
y = Yeff
ux = x*np.sqrt((uXeff/Xeff)**2)
#ux = x*np.sqrt((0.1/f)**2+(uXeff/Xeff)**2+(uR/R)**2)
uy = uYeff
def f(x,p):
    "Formule de Borda "
    M = p
    return M*x

# derivee de la fonction f par rapport à la variable de controle x
def Dx_f(x,p):
    M =p
    return M

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
#print(chi2r)

print(r"On mesure un coefficient d'induction mutuelle M = " + str(np.round(popt[0],8))+ "+/- "+ str(np.round(upopt[0],7)) + " H " )
print(r"Le chi2 est de " + str(np.round(chi2r,3)))
#%%
plt.figure()
#plt.plot(didt,Yeff, 'o')
plt.errorbar(didt, Yeff, xerr= ux, yerr = uy,fmt= 'o')
abscisse = np.linspace(20,110,1000)
plt.plot(abscisse, f(abscisse, popt))
plt.xlabel(r'$\frac{di_1}{dt}$ (A/s)')
plt.ylabel(r'$e_{ind}$ (V)')


