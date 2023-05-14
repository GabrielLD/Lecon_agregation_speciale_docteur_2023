
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 08:56:19 2015

@author: ENS de Lyon (titre original : Snanon vision temporelle)

2018 : modifiÃ© par S.Dorizon prÃ©pa agreg Rennes
ajout d'une figure avec le spectre des deux signaux

Objectif : Illustre le thÃ©orÃ¨me de Shanon et le repliement spectral en
comparant un signal rÃ©el et un signal acquis.

EntrÃ©es :   - f : frÃ©quence du signal Ã  acquÃ©rir
            - N : nombre de points du signal Ã  acquÃ©rir ()
            - fe : frÃ©quence d'Ã©chantillonnage
            - T : durÃ©e de l'acquisition
            
Sortie :    - figure 1 : comparaison du signal acquis et du signal Ã  acquÃ©rir
            - figure 2 : spectre du signal Ã  acquÃ©rir (symÃ©trique par rapport Ã  0)
            -figure 3 : spectre du signal Ã©chantillonnÃ©
"""

import numpy as np
import matplotlib.pyplot as plt

f = 100# FrÃ©quence du signal Ã  acquÃ©rir
fe = 150# FrÃ©quence d'Ã©chantillonage
"""
attention ne pas prendre un rapport entier entre f et fe
choisir par exemple 100 et 133 ou 100 et 247
"""

T = 0.2# Duree de l'acquisition
N=150
N1=fe*T
print(N)

t = np.linspace(0,T,N)# Temps pour le signal Ã  acquÃ©rir

Y = np.sin(2*np.pi*f*t)# Signal Ã  acquÃ©rir
YTF = np.fft.fftshift(np.fft.fft(Y))
fr= np.fft.fftshift(np.fft.fftfreq(N,T/N))
#YTF=np.fft.fft(Y);

te = np.linspace(0,T,int(T*fe))# Temps pour le signal acquis
Ye = np.sin(2*np.pi*f*te)# Signal acquis
YeTF= np.fft.fftshift(np.fft.fft(Ye));
frr= np.fft.fftshift(np.fft.fftfreq(int(fe*T),1/(fe)))

plt.subplot(311)
plt.plot(t,Y,label="Signal reel")# Affichage du signal Ã  acquÃ©rir
plt.plot(te,Ye,'o-',label="Signal acquis")# Affichage du signal acquis
#plt.plot(te,Ye,label="Signal acquis")# Affichage du signal acquis
plt.legend()
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude du signal")
plt.title("Comparaison du signal rÃ©el et du signal acquis")


plt.subplot(312)
plt.plot(fr,abs(YTF)/N,color="blue",label="Signal rÃ©el")

#plt.subplot(313)
plt.plot(frr,abs(YeTF)/N1,color="red",label="Signal acquis")
plt.legend()
plt.xlabel("freq (Hz)")
plt.ylabel("Amplitude")

plt.show()