
"""Orbites de Keppler

Description
-----------
Ce programme permet d'illustrer les orbites de deux astres, dont les masses peut être changées. 
En rouge et bleu sont tracées les trajectoires des deux astres. L'excentricité e, 
le demi-grand axe a et l'angle initial theta_0 peuvent être changés.

Informations
------------
Auteurs : François Lévrier, Arnaud Raoux et la prépa agreg de Montrouge
Année de création : 2016 
Version : 1.2
Licence : Creative Commons Attribution - Pas d'utilisation Commerciale 4.0 International

Liste des modifications:
    * v 1.00 : 2016-05-15 Première version complète
    * v 1.10 : 2019-01-09 Remplacement de axisbg dépréciée par facecolor
    * v 1.2 : 2019-05-19

"""
import numpy as np
from numpy.lib.scimath import sqrt
import matplotlib.pyplot as plt

from programmes_lecons import FloatSlider, IntSlider
from programmes_lecons import make_param_widgets, make_choose_plot, make_reset_button
from programmes_lecons import justify
from programmes_lecons.constantes import *

titre = "Orbites de Keppler"

description = """Ce programme permet d'illustrer les orbites de deux astres, dont les masses peut être changées. 
En rouge et bleu sont tracées les trajectoires des deux astres. L'excentricité e, 
le demi-grand axe a et l'angle initial theta_0 peuvent être changés."""

#===========================================================
# --- Variables globales et paramètres ---------------------
#===========================================================

parameters = dict(
    theta_0 = FloatSlider(description=r'$\theta_0$', min=0.0, max=180.0, value=0),
    mass_ratio = FloatSlider(description = r'$\log_{10}(M_1/M_2)$', min=-2, max=2, value=1),
    a = FloatSlider(description = r'Demi-grand axe -- $a$ (en UA)', min=0.1, max=10.0, value=1),
    e = FloatSlider(description = r'Excentricité -- $e$', min=0.0, max=0.9999999, value=0),
)


#===========================================================
# --- Modèle physique --------------------------------------
#===========================================================

theta=np.linspace(0.000001,2.0*np.pi-0.000001, num=1000)

def trace(theta0, M1, M2, a, e): # Tout doit être en SI, rad
    # Masse reduite
    mu = M1*M2/(M1+M2)
    # Facteur K de la force centrale F=-K/r2
    K = G*M1*M2
    L = np.sqrt(K*mu*a*(1.0-e**2))
    p = L**2/(K*mu)
    # Mouvement de la particule fictive
    r = p/(1+e*np.cos(theta-theta0))
    # Projection de ce mouvement
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    # Mouvement de l'objet M1
    x1 = -(M2/(M1+M2))*x
    y1 = -(M2/(M1+M2))*y
    # Mouvement de l'objet M2
    x2 = (M1/(M1+M2))*x
    y2 = (M1/(M1+M2))*y
    return x, y, x1, y1, x2, y2


#===========================================================
# --- Réalisation du plot ----------------------------------
#===========================================================

# La fonction plot_data est appelée à chaque modification des paramètres
def plot_data(theta_0, mass_ratio, a, e):
    theta_0 = theta_0/180*np.pi
    a = a*UA
    M2 = Msun
    M1 = 10**(mass_ratio)*M2

    x, y, x1, y1, x2, y2 = trace(theta_0, M1, M2, a, e)

    lines['Masse réduite'].set_data(x/UA, y/UA)
    lines['M1'].set_data(x1/UA, y1/UA)
    lines['M2'].set_data(x2/UA, y2/UA)


#===========================================================
# --- Création de la figure et mise en page ----------------
#===========================================================

fig = plt.figure()
fig.suptitle(titre)
fig.text(0.02, .9, justify(description), multialignment='left', verticalalignment='top')

ax = fig.add_axes([0.35, 0.25, 0.6, 0.65], aspect='equal')

lines = {}
lines['Masse réduite'], = ax.plot([], [], 'k-', lw=2, visible=False)
lines['M1'], = ax.plot([], [], 'r-',lw=2)
lines['M2'], = ax.plot([], [], 'b-',lw=2)

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

param_widgets = make_param_widgets(parameters, plot_data, slider_box=[0.45, 0.07, 0.5, 0.07])
reset_button = make_reset_button(param_widgets)
choose_widget = make_choose_plot(lines, box=[0.015, 0.10, 0.2, 0.15])

if __name__=='__main__':
    plt.show()


