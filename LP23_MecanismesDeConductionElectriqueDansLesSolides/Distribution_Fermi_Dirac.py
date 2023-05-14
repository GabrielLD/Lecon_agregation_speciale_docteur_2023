# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 18:19:56 2023

@author: Gabriel

FERMI DIRAC
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


##### (1) DÃ©finition de la fonction
def ma_fonction(x):
    y = np.zeros(x.size)
    for i in range(x.size):
        y = 1/(1+(x+B)/(k*A))
        #y = A*np.sin(x*B*np.pi)
    return y

k = 3
##### (2) DÃ©finition des paramÃ¨tres A et B
# Valeur initiale
A0 = 1
B0 = .5
A = A0
B = B0
# valeur extremales
Amin = 0
Amax = 100*A0
Bmin = 0
Bmax = 2*A0
# Nom des paramÃ¨tres
Anom = "Température"
Bnom = "densite volumique d electrons"


##### DÃ©finition de l'axe des x 
# N est l nombre de points sur l'axe
N = 101
x = np.linspace(0,4,N)

##### DÃ©finition du graphique
fig = plt.figure() #figure animation
# ax=plt.axes(xlim=(0,1.),ylim=(-1.2,2.05))  #Axes x et y
ax = plt.axes()
plt.subplots_adjust(left = 0.1,bottom=0.25)  #on place le graphique sur la page
courbe, = ax.plot(x,ma_fonction(x))
# premier slider
A_axSlider = plt.axes([0.2,0.07,0.7,0.05])
A_Slider = Slider(A_axSlider,Anom,Amin,Amax,A0)  
# deuxiÃ¨me slider
B_axSlider = plt.axes([0.2,0.12,0.7,0.05])
B_Slider = Slider(B_axSlider,Bnom,Bmin,Bmax,B0)

##### fonction pour modifier les paramÃ¨tres et actulaiser la courbe
def update(val):    
    global A, B
    # on change la valeur des paramÃ¨tres
    A = A_Slider.val
    B = B_Slider.val
    # on recalcul et on affiche la fonction
    courbe.set_ydata(ma_fonction(x))
    
A_Slider.on_changed(update)
B_Slider.on_changed(update)

# On lance le calcul du graph
plt.show()
