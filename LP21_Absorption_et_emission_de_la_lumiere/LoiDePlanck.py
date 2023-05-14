"Loi de PLanck"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

h = 6.626e-34  #Constante de plank
c = 299792458    #Vitesse lumiÃ¨re
k = 1.38e-23  #Constante de Boltzman

def planck(Lambda):
    a = (2*np.pi*h*c**2)/(Lambda**5)
    b = h*c/(Lambda*k*T)
    intensite_P = a/(np.exp(b)-1.0)
    intensite_W = (8*np.pi*h*c**2)/(4*Lambda**5)*np.exp(-b)
    intensite_R = (8*np.pi*k*T*c)/(4*Lambda**4)
    return intensite_P,intensite_W,intensite_R

# Valeur initiale
T0 = 6000
T = T0

# valeur extremales
Tmin = 0
Tmax = 2*T0

# Nom des paramÃ¨tres
Tnom = "TempÃ©rature"


#lumiÃ¨re visible (micromÃ¨tre)
a=0.38*10**-6
b=0.78*10**-6

x = np.arange(1e-9, 1e-3, 10e-9)

fig = plt.figure() #figure animation

ax = plt.axes()
plt.subplots_adjust(left = 0.1,bottom=0.2)  #on place le graphique sur la page
courbeP, = ax.plot(x,planck(x)[0],linewidth=5, color="black",label="Planck")
courbeW, = ax.plot(x,planck(x)[1],"r--",linewidth=3, color="blue",label="Wien")
courbeR, = ax.plot(x,planck(x)[2],"r--",linewidth=3, color="red",label="Rayleigh-Jeans")

# premier slider
T_axSlider = plt.axes([0.2,0.07,0.7,0.05])
T_Slider = Slider(T_axSlider,Tnom,Tmin,Tmax,T0)


def update(val):
    global T
    # on change la valeur des paramÃ¨tres
    T = T_Slider.val
    # on recalcul et on affiche la fonction
    courbeP.set_ydata(planck(x)[0])
    courbeW.set_ydata(planck(x)[1])
    courbeR.set_ydata(planck(x)[2])

T_Slider.on_changed(update)

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim(1e1, 1e+15)

ax.set_ylabel("Ã©mittance spectrale",fontsize=20, color="darkblue")
ax.set_xlabel("Longueur d'onde en mÃ¨tres",fontsize=20, color="darkblue")
ax.set_title("Rayonnement  ",fontsize=20, color="blue")

ax.legend(fontsize=15)

rainbow_colors = ['violet', 'indigo', 'blue', 'green', 'yellow', 'orange', 'red']
size_color = (b - a) / len(rainbow_colors)
for index, color in enumerate(rainbow_colors):
    start_for_new_color = a + index*(b-a) / len(rainbow_colors)
    end_for_new_color = start_for_new_color + size_color
    xs = np.linspace(start_for_new_color, end_for_new_color, 100)
    ax.vlines(xs, 10e0, 10e14, color=color)

# On lance le calcul du graph
ax.grid()
plt.show()