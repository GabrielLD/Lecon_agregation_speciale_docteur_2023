
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 14:39:50 2017

@author: ENS de Lyon

Objectif : On part d'un signal comportant du bruit ou tout simplement des frÃ©quences non dÃ©sirÃ©es.
On regarde le spectre du signal et on applique un filtre passe bas pour supprimer les frÃ©quences non voulues.
On calcule alors le signal thÃ©orique qu'on devrait obtenir aprÃ¨s le filtre pour montrer qu'on a bien diminuÃ©
les hautes frÃ©quences qui nous gÃªnaient. Le signal est un cosinus amorti et le bruit aussi dont on controle la
 frÃ©quence et facteur de qualitÃ©

EntrÃ©es:
        -T = temps du signal (le prendre grand pour avoir une FFT rÃ©solu) en seconde
        -N = nombre de points dans le signal
        -f1 = basse frÃ©quence que l'on souhaite garder en hertz
        -f2 = haute frÃ©quence que l'on souhaite supprimer en hertz
        -Q1 = facteur de qualitÃ© de la basse frÃ©quence
        -Q2 = facteur de qualitÃ© de la haute frÃ©quence
        -fc = frÃ©quence de coupure du filtre passe bas en hertz

Sortie : On obtient une figure contenant 5 graphes : en haut Ã  gauche le signal et en haut Ã  droite le signal
 filtrÃ©, en bas Ã  gauche la FFT du signal et en bas Ã  droite la FFT du signal filtrÃ©, au milieu en bas la
 fonction de transfert du filtre passe bas

"""

import numpy as np
import matplotlib.pyplot as plt


#entrÃ©es :
T=2 # en s
N=100000
f1=100#en Hz
f2=1000# en Hz
Q1=10
Q2=200
fc = 100 # en Hz


t = np.linspace(0,T,N)# on crÃ©e la ligne de temps
#on crÃ©e le signal
signal = np.exp(-np.pi*f1/Q1*t)*np.cos(2*np.pi*f1*t)+np.exp(-np.pi*f2/Q2*t)*np.cos(2*np.pi*f2*t)
#on calcule la FFT du signal et les frÃ©quences
TF = np.fft.fftshift(np.fft.fft(signal))
fr= np.fft.fftshift(np.fft.fftfreq(N,T/N))

plt.figure()
plt.clf()
# on trace le signal
plt.subplot(2,3,1)
plt.plot(t,signal)
plt.xlim([0,0.1]) # le signal pour les paramÃ¨tres initiaux est compris entre 0 et 0.1 seconde
plt.xlabel('Temps (s)')
plt.ylabel('Signal')
plt.title('Signal')
# on trace la FFT du signal
plt.subplot(2,3,4)
plt.plot(fr,np.abs(TF))
plt.xlim([0,1100]) # pour les paramÃ¨tres initiaux, les frÃ©quences intÃ©ressantes sont infÃ©rieure Ã  1100 Hz
plt.xlabel('FrÃ©quence (Hz)')
plt.ylabel('TF du signal')
plt.title('TF du Signal')


# on trace la fonction de transfert du filte
plt.subplot(2,3,5)
H = 1/(1+np.complex(0,1)*fr/fc) # Calcul de la fonction de transfert pour un filtre PB d'ordre 1
plt.plot(fr,np.abs(H))
plt.xlim([0,1100]) # on garde le mÃªme affichage que la FFT du signal
plt.xlabel('Frequence (Hz)')
plt.ylabel('Fonction de transfert du filtre')
plt.title('Filtre passe bas de frequence de coupure {}Hz'.format(fc))

# On trace la FFT du signal filtrÃ©
plt.subplot(2,3,6)
TFfiltre= [TF[i]*H[i] for i in range(len(fr))]# on calcule la FFT du signal filtrÃ© en multipliant la FFT du signal par la fonction de transfert
plt.plot(fr,np.abs(TFfiltre))
plt.xlim([0,1100])# on garde le mÃªme affichage pour rester logique
plt.xlabel('Frequence (Hz)')
plt.ylabel('TF du signal filtre')
plt.title('TF de Signal filtre')

#on trace le signal filtrÃ©
plt.subplot(2,3,3)
signalfiltre = np.real(np.fft.ifft(np.fft.fftshift(TFfiltre)))#on calcule le signal filtrÃ© Ã  partir de sa FFT
plt.plot(t,signalfiltre)
plt.xlim([0,0.1])# on garde le mÃªme affichage que pour le signal
plt.xlabel('Temps (s)')
plt.ylabel('Signal filtre')
plt.title('Signal filtre')


plt.show()





