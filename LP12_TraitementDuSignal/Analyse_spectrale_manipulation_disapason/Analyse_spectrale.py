#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 16:19:35 2022

@author: thibault
"""

#-----------------------------------------------------------------------
# Analyse spectrale d'un signal issu du diapason
#-----------------------------------------------------------------------

# Bibliothèques utilisées

import numpy as np
import numpy.fft as fft
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')
# import matplotlib.animation as ani
# import matplotlib.widgets as mwg

from numpy import array

#-----------------------------------------------------------------------
# Charger les données à analyser

s_2000=np.loadtxt(open("diapason_seul_10000.csv"), delimiter=",", skiprows=2)
# array of data
data_s_2000=array(s_2000)

# Calcul du spectre

# N = 8192
# T = np.linspace(0.0, 1.0, N)
# Y = foo(T)
spectre_s_2000 = fft.fft(data_s_2000[:,1])
freq_s_2000 = fft.fftfreq(len(data_s_2000[:,0]),data_s_2000[1,0]-data_s_2000[0,0])

print(len(spectre_s_2000))

# Tracés

# Tr = np.linspace(0.0, 2.0, 2*N)
# Ampl = 20*np.log10(np.abs(Yp[:max_harm]))

# Signal

# axTmp = plt.axes([0.11, 0.6, 0.78, 0.32])
fig,ax = plt.subplots()
plt.plot(data_s_2000[:,0],data_s_2000[:,1], "b--")
# partial, = axTmp.plot(Tr, np.concatenate([Y, Y]), "b")
# axTmp.set_xlim([0, 2])
# axTmp.set_ylim([ min(Y) - (max(Y)-min(Y))*0.3, max(Y) + (max(Y)-min(Y))*0.3 ])
plt.title("signal")
plt.ylabel("amplitude (V)")
plt.xlabel("temps (s)")
# inset axes....
axins = ax.inset_axes([0.5, 0.7, 0.47, 0.27])
# axins.imshow(Z2, extent=extent, origin="lower")
# sub region of the original image
x1, x2, y1, y2 = 5, 5.01, -.2, .2
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.plot(data_s_2000[:,0],data_s_2000[:,1], "b--")

ax.indicate_inset_zoom(axins, edgecolor="black")
plt.show()

# Spectre

n_split=int(np.floor(len(spectre_s_2000)/2))
# axTmp = plt.axes([0.11, 0.6, 0.78, 0.32])
plt.plot(freq_s_2000[0:n_split],np.abs(spectre_s_2000[0:n_split]), "b-")
# partial, = axTmp.plot(Tr, np.concatenate([Y, Y]), "b")
plt.xlim(0,max(freq_s_2000))
# axTmp.set_ylim([ min(Y) - (max(Y)-min(Y))*0.3, max(Y) + (max(Y)-min(Y))*0.3 ])
plt.title("spectre")
plt.ylabel("amplitude (dB)")
plt.xlabel("fréquence (Hz)")
plt.show()