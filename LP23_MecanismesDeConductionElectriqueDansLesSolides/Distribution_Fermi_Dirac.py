# -*- coding: utf-8 -*-
"""
Created on Wed May 24 23:03:50 2023

@author: Gabriel
"""

import matplotlib.pyplot as plt
import numpy as np
#import matplotlib.animation as anim    
from matplotlib.widgets import Slider, Button, RadioButtons


e= np.linspace(0,10,1000)
mu = 2
x = e/mu
kT = mu/200

FD = 1/(1+np.exp((x-mu)/kT))
plt.figure()
plt.plot(x, FD)

def ma_fonction(x):
    y = np.zeros(x.size)
    for i in range(x.size):
        y = 1/(1+np.exp((x-mu)/A))
    return y
A0 = mu/2
A = A0
#VALEUR EXTREMALES

Amin = A0/100
Amax = 2*A0
Anom = "kT"

fig = plt.figure() #figure animation
# ax=plt.axes(xlim=(0,1.),ylim=(-1.2,2.05))  #Axes x et y
ax = plt.axes()
plt.subplots_adjust(left = 0.1,bottom=0.25)  #on place le graphique sur la page
courbe, = ax.plot(x,ma_fonction(x))
ax.plot(x,FD, label = 'T=0 K')
# premier slider
A_axSlider = plt.axes([0.2,0.07,0.7,0.05])
A_Slider = Slider(A_axSlider,Anom,Amin,Amax,A0)  
ax.legend()

##### fonction pour modifier les paramÃ¨tres et actulaiser la courbe
def update(val):    
    global A
    # on change la valeur des paramÃ¨tres
    A = A_Slider.val
    # on recalcul et on affiche la fonction
    courbe.set_ydata(ma_fonction(x))
    
A_Slider.on_changed(update)
# On lance le calcul du graph
plt.show()