# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:25:44 2023

@author: Gabriel
"""

"""
Ce programme permet de visualiser les fonctions d'onde des Ã©tats liÃ©s dans un
un puits quantique fini.
On pourra voir l'Ã©volution de ces fonctions d'ondes suite Ã  l'Ã©largissement du
puits et de sa profondeur

les equations qui lient les vecteurs d'onde et les coeffiecients d'attenuation sont:
   un cercle d'Ã©quation:
       (ka)Â²+(qa)Â²=2mVaÂ²/hbarÂ²
   pour les Ã©tats paires
       qa=ka*tan(ka)
   pour les Ã©tat impaires
       qa=-ka*cotan(ka)

En cherchant l'intersection entre le cercle et les deux autres fonctions, on
obtient les valeurs des vecteurs d'onde.
Avec l'Ã©quation des Ã©tats on remonte aux coeficients d'attenuation des ondes
Ã©vanescentes.

Les Ã©quation des fonctions d'onde dans le puit sont de la forme

Y=A*cos(kx) pour les Ã©tats paires et
Y=A*sin(kx) pour les Ã©tats impaires

Les equations des ondes evanescente ont des Ã©quations de la forme

Y=B*exp(-qx) pour les x positif et
Y=B*exp(qx) pour les x nÃ©gatif
"""

import matplotlib
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons

# DÃ©termination des vecteur d'onde k et des coefficient q d'attenuation
# des ondes Ã©vanescentes

R2= lambda V,a : 2*m*V*a*a/(hbar*hbar)

def tangente(x):
   """
   Cette fonction reprÃ©sente les etats paires dans le puits, deduits a  partir
   de la continuite de la fonction d'onde au bord du puits.
   x=k*a
   """
   return(x*np.tan(x))


def cotangente(x):
   """
   Cette fonction reprÃ©sente les Ã©tats inpaires dans le puits, dÃ©duits Ã  partir
   de la continuite de la fonction d'onde au bord du puits.
   x=k*a
   """
   return(-x*np.cos(x)/np.sin(x))


def cercle(r2,x):
   """
   Cette fonction represente le lien entre les vecteurs d'ondes k des etats lies
   et de leur ondes evanescentes en fonction des paramÃ¨tres du puits.
   x=k*a
   """
   return(np.sqrt(r2-x*x))


def fsym(x, V, a): #soustraction des deux fonctions
   """
   input:
      a est la largeur du puits,
      V est la hauteur du puits (c'est le potentiel)
   """
   r2=R2(V,a)
   y= tangente(x)-cercle(r2,x)
   return(y)

def fantisym(x, V, a): #soustraction des deux fonctions
   """
   input:
      a est la largeur du puits,
      V est la hauteur du puits (c'est le potentiel)
   """
   r2=R2(V,a)
   y= cotangente(x)-cercle(r2,x)
   return(y)



def val_sol(fsym,fantisym,e,V,a):
   '''
   Cette fonction cherche l'intersection entre les fonctions des Ã©tats paires
   et impaires avec la fonction cercle qui reprÃ©sente le caractÃ¨re
   intrinsÃ¨que du puits.
   input:
          a:   la largeur du puits
          V:   la profondeur
          e:   l'erreur sur la valeur du vecteur d'onde
   output:
       on obtient une liste de x pour lequel une fonction d'onde existe
   '''
   i=0
   l=np.array([])

   while i<np.sqrt(R2(V,a)):
       A=i+e/10
       B=min(i+np.pi/2-e/10,np.sqrt(R2(V,a)))
       if fsym(A,V,a)*fsym(B,V,a)>0:
           print("pas de solution")
           return(l)
       else :
           while B-A>=e:
               C=(A+B)/2
               if fsym(A,V,a)*fsym(C,V,a)<=0:
                   B=C
               else :
                   A=C
       l=np.append(l,A)

       if i+np.pi/2<np.sqrt(R2(V,a)):
           A=i+np.pi/2+e/10
           D=min(i+np.pi-e/10,np.sqrt(R2(V,a)))
           if fantisym(A,V,a)*fantisym(D,V,a)>0:
               print("pas de solution")
               return(l)
           else :
               while D-A>=e:
                   C=(A+D)/2
                   if fantisym(A,V,a)*fantisym(C,V,a)<=0:
                       D=C
                   else :
                       A=C
           l=np.append(l,D)
       i=i+np.pi
   return l

def listek(fsym,fantisym,e,V,a):
   """
   Comme x=k*a on divise tout les valeur de x par a pour retrouver nos
   vecteur d'onde k
   input:

          a:   la largeur du puits
          V:   la profondeur
          e:   l'erreur sur la valeur du vecteur d'onde
   output:
       liste des vecteurs d'onde k corespondants Ã  tous les Ã©tats liÃ©s du puits
   """
   k=val_sol(fsym,fantisym,e,V,a)/a
   return(k)

def T1(a=10e-9,V=8e-19,e=1e-5):
   '''
      Cette fonction calcule les valeurs des diffÃ©rentes fonctions d'onde
      input :
            a:   la largeur du puits
            V:   la profondeur
            e:   l'erreur sur la valeur du vecteur d'onde
      output:
          liste de valeurs pour tracer les fonctions
   '''
   output= list()
   lx=np.linspace(-a,a,1000)
   lx1=np.linspace(a,1.5*a,1000)
   lx2=np.linspace(-1.5*a,-a,1000)
   k=listek(fsym,fantisym,e,V,a)
   q=tangente(k*a)/a
   E=(hbar*hbar*k*k)/(2*m)
   for i,j,s in zip(k,E, range(len(E))):
       if s%2==0:
           q=tangente(i*a)/a
           ly= np.cos(lx*i)
           ly1=np.exp(-q*(lx1-a))*ly[-1]
           ly2=np.exp(q*(lx2+a))*ly[0]
       else:
           q=cotangente(i*a)/a
           ly=np.sin(lx*i)
           ly1=np.exp(-q*(lx1-a))*ly[-1]
           ly2=np.exp(q*(lx2+a))*ly[0]
       output.append(np.hstack([ly2,ly,ly1]))
   return np.hstack([lx2,lx,lx1]), np.array(output)

# initialisation des variables
m=9.10E-31
hbar=1.0546E-34
e=0.0001



def plot_ani(a=2E-9, V=8E-19, e=0.0001):
   '''
      Cette fonction trace les fonctions d'onde des Ã©tats
      input :
            a:   la largeur du puits
            V:   la profondeur
            e:   l'erreur sur la valeur du vecteur d'onde
      output:
          tracer des fonctions d'ondes et de leurs ondes Ã©vanescentes
   '''

   fig, ax=plt.subplots(1,2)
   plt.subplots_adjust(left=0.15, bottom=0.35)
   axcolor='red'
   axV = plt.axes([0.15, 0.07,  0.75, 0.03], facecolor=axcolor)
   axa = plt.axes([0.15, 0.15, 0.75, 0.03], facecolor=axcolor)

   #tracer des 3 fonctions pour dÃ©terminer graphiquement les solutions dans la premiÃ¨re fenÃªtre
   Lx=np.linspace(0,np.sqrt(R2(V,a)),1000)
   Ly= tangente(Lx)
   Ly2= cotangente(Lx)
   Ly3=np.sqrt(R2(V,a)-Lx*Lx)
   ax[0].plot(Lx,Ly3,'g')
   ax[0].plot(Lx,Ly2,'b')
   ax[0].plot(Lx,Ly,'b')
   ax[0].axis([0,np.sqrt(R2(V,a)),0,35])


   for i in val_sol(fsym,fantisym,e,V,a):
       ax[0].plot([i,i],[0,np.sqrt(R2(V,a)-i*i)],'red', 2.5,"--")
       ax[0].scatter([i,],[np.sqrt(R2(V,a)-i*i),], 50, color ='red')



   #tracer du puits et des fonctions d'onde dans la deuxiÃ¨me fenÃªtre

   pot_wall=[[-1.5*a,-a,-a,a, a,1.5*a],[0, 0, -V ,-V, 0,0] ]
   #pot_wall=[[-1.5*a/2,-a/2,-a/2,a/2, a/2,1.5*a/2],[0, 0, -V ,-V, 0,0] ]
   
   ax[1].plot(pot_wall[0],pot_wall[1], 'k')
   x,out=T1(a=a,V=V,e=1e-5)
   for i,funci in enumerate(out):
       const=V/len(out)*0.5
       ax[1].plot(x, funci*const -V +i* V/len(out))


   # crÃ©ation de crurseur pour choisir manuellement la profondeur du puits et sa largeur

   sl_axV = Slider(axV, 'V (E-19 C)', 0.001, 1.0, valinit=0.1)
   sl_axa = Slider(axa, 'a (E-9 m)', 0.1, 5.0, valinit=2)

   def update(val):
       """
       fonction permettant de re-tracer tous les graphiques suite Ã  un
       changement des variables d'entrÃ©e
       """
       V = sl_axV.val*1E-19
       a = sl_axa.val*1E-9
       ax[0].clear()
       ax[1].clear()
       Lx=np.linspace(0,np.sqrt(R2(V,a)),1000)
       Ly= tangente(Lx)
       Ly2= cotangente(Lx)
       Ly3=np.sqrt(R2(V,a)-Lx*Lx)
       ax[0].plot(Lx,Ly3,'g')
       ax[0].plot(Lx,Ly2,'b')
       ax[0].plot(Lx,Ly,'b')
       ax[0].axis([0,np.sqrt(R2(V,a)),0,35])
       for i in val_sol(fsym,fantisym,e,V,a):
           ax[0].plot([i,i],[0,np.sqrt(R2(V,a)-i*i)],'red', 2.5,"--")
           ax[0].scatter([i,],[np.sqrt(R2(V,a)-i*i),], 50, color ='red')
       x,out=T1(a=a,V=V,e=1e-5)

       pot_wall=[[-1.5*a,-a,-a,a, a,1.5*a],[0, 0, -V ,-V, 0,0] ]
       ax[1].plot(pot_wall[0],pot_wall[1], 'k')
       for i,funci in enumerate(out):
           const=V/len(out)*0.5
           ax[1].plot(x, funci*const -V +i* V/len(out))


       fig.canvas.draw_idle()

   sl_axV.on_changed(update)
   sl_axa.on_changed(update)

   update(0)
   plt.show()

if __name__ =='__main__':
   plot_ani()


# Note pour la correction
# cotangente(x) peut diverger
#   => il faudrait s'assurer que x n'est aps un multiple de pi/2
# fsym est une fonction
#    mais elle apparait en argument de plein d'autres fonction ???
# autres erreurs :
#  L66 cercle a une racine d'un negatif
#  L257 idem


