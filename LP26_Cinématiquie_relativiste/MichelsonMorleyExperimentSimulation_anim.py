# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 23:11:47 2023

@author: GLD
"""

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as anim    
from matplotlib.widgets import Slider, Button, RadioButtons

"""
Principe :
    Le programme trace une fonction Ã  l'aide de matplotlib
    Il ajoute des "widget" "Slider" qui permettent de modifier
    deux paramÃ¨tres de la fonction.

Mode d'emploi :
    (1) Mettre la fonction voulue dans "ma_fonction"
        Les deux paramÃ¨tres sont notÃ©s A et B.
    (2) Rentrer les valeurs initiales, min et max
        ainsi que le nom des paramÃ¨tres.


"""
lambda_ = 550E-9 # m
ve = 30E3 #m/s
c = 3E8 # m/s
L1 = 11 #m
L2 = 11 #m


def Phi(ve, L1, L2, lambda_): 
    beta = ve/c
    # Interferometre a 0 deg
    phi = 2*np.pi*(2*(L2-L1)+(2*L2-L1)*beta**2)/lambda_
    # interferometre tourne de 90 deg
    phip = 2*np.pi*(2*(L2-L1)+(L2-2*L1)*beta**2)/lambda_
    Phi = phip-phi
    return Phi

def michelson_morley(x,ve, L1, L2, lambda_):
    I0 = 1
    I = 2*I0*(1+np.cos(x+Phi(ve, L1, L2, lambda_)))
    return I



##### (1) DÃ©finition de la fonction
def ma_fonction(x):
    I0 = 1
    L1 = 11
    L2 = 11
    lambda_ = 550E-9
    phi = 2*np.pi*(2*(L2-L1)+(2*L2-L1)*A**2)/lambda_
    phip = 2*np.pi*(2*(L2-L1)+(L2-2*L1)*A**2)/lambda_
    Phi = phip-phi
    I = np.zeros(x.size)
    for i in range(x.size):
        I = 2*I0*(1+np.cos(x+Phi))
    return I


##### (2) DÃ©finition des paramÃ¨tres A et B
# Valeur initiale
A0 = ve/c
#B0 = 1
A = A0
#B = B0
# valeur extremales
Amin = 1/c
Amax = 1
#Bmin = 0
#Bmax = 2*A0
# Nom des paramÃ¨tres
Anom = "vitesse"
#Bnom = "vecteur d'onde"


##### DÃ©finition de l'axe des x 
# N est l nombre de points sur l'axe
N = 1001
x = np.linspace(0,10,N)

##### DÃ©finition du graphique
fig = plt.figure() #figure animation
# ax=plt.axes(xlim=(0,1.),ylim=(-1.2,2.05))  #Axes x et y
ax = plt.axes()
plt.subplots_adjust(left = 0.1,bottom=0.25)  #on place le graphique sur la page
courbe, = ax.plot(x,ma_fonction(x), label = 'mesure de Michelson et Morley')
ax.plot(x,michelson_morley(x,c, L1, L2, lambda_), label ='vitesse c constante')
plt.legend(loc = 'upper right')
plt.xlabel('distance')
plt.ylabel('I')
# premier slider
A_axSlider = plt.axes([0.2,0.07,0.7,0.05])
A_Slider = Slider(A_axSlider,Anom,Amin,Amax,A0)  
# deuxiÃ¨me slider
#B_axSlider = plt.axes([0.2,0.12,0.7,0.05])
#B_Slider = Slider(B_axSlider,Bnom,Bmin,Bmax,B0)

##### fonction pour modifier les paramÃ¨tres et actulaiser la courbe
def update(val):    
    global A#, B
    # on change la valeur des paramÃ¨tres
    A = A_Slider.val
    #B = B_Slider.val
    # on recalcul et on affiche la fonction
    courbe.set_ydata(ma_fonction(x))
    
A_Slider.on_changed(update)
#B_Slider.on_changed(update)

# On lance le calcul du graph
plt.show()