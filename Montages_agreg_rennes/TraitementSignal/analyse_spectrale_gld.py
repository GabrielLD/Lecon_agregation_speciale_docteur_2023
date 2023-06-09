#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 08:49:56 2023

@author: gld

Traitement du signal
"""


#%% PARAMETRES D'ACQUISITION

"""
Acquisition du signal avec Oscilloscope pour fixer les paramètres d'acquisition puis sous latispro

Temps d'acquisition : 
Fréquence d'acquisition :10 000
Amplitude max du signal : 50 mV


"""

#%%
import numpy as np
import numpy.fft as fft
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
import pandas as pd
from numpy.fft import fft, fftfreq

#dfs = pd.read_csv('Diapason440Hz_Points150000Te20usTotal3s.csv', sep = ';')
dfs = pd.read_csv('Diapason440Hz_Points60000Te50usTotal3s.csv', sep = ';')
Te = 50e-6
s = dfs['EA4'].to_numpy()
t = dfs['Temps'].to_numpy()


plt.figure()
plt.plot(t,s, '+ ' )
plt.xlabel('time (s)')
plt.ylabel('s (V)')

#%% SPECTRE

spectre_s = fft(s)  # Transformée de fourier
freq_s = fftfreq(s.size, d=t[1]-t[0])  # Fréquences de la transformée de Fourier

plt.figure()
plt.plot(freq_s,spectre_s.real, "b-")
plt.plot(freq_s,spectre_s.imag, "r-")
#plt.xscale('log')


plt.figure()
plt.plot(freq_s,spectre_s.real, "b-")
plt.plot(freq_s,spectre_s.imag, "r-")
plt.xlim([410,450])

plt.title("spectre")
plt.ylabel("amplitude (dB)")
plt.xlabel("fréquence (Hz)")
plt.show()

#%% Étude de l'échantillonage d'un signal

ech=20

t2 = t[::ech]
s2 = s[::ech]
spectre_s2 = fft(s2)  # Transformée de fourier
freq_s2 = fftfreq(s2.size, d=t[ech]-t[0])  # Fréquences de la transformée de Fourier


plt.figure()
plt.plot(t,s, '+g',  label = 'signal acquis')
plt.plot(t2, s2, 'o', label = 'signal sous échantilloné')

plt.figure()
plt.plot(freq_s,np.abs(spectre_s), "g-")
plt.plot(freq_s2, np.abs(spectre_s2), 'b')
plt.xlim([0,450])