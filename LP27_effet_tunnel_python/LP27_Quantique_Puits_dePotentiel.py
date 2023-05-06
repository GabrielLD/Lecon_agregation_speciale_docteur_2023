

"""
Le programme ci-dessous permet la rÃ©solution numÃ©rique de l'Ã©quation de SchrÃ¶dinger dÃ©pendante du temps, appliquÃ©e sur un
paquet d'ondes diffusÃ© par un potentiel.
"""


"""
On dÃ©finit et on appelle les bibliothÃ¨ques python que nous utiliserons dans la suite :

- Numpy : calculs matriciels, dÃ©finitions de fonctions mathÃ©matiques (exp,sin) et de nombres complexes (j)
- Pyplot : tracÃ© de graphiques
- Animation : animation de courbes
- Widgets : Curseurs et boutons pour pouvoir modifier le tracÃ© des courbes en direct

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim    
from matplotlib.widgets import Slider, Button, RadioButtons


"""
PRELIMINAIRE : On impose les variables de base de notre systÃ¨me.
               On utilise les unitÃ©s atomiques.
"""

#Valeurs liÃ©es au temps et Ã  l'espace

n = 1000 #nombre de points dans l'espace
p = 500 #nombre d'itÃ©rations dans le temps
delta_t = 0.000001 #pas de temps
delta_x = 0.001 #pas d'espace

#Valeurs liÃ©es Ã  la fonction d'onde initiale

sigma_init = 0.04 #valeur de sigma initiale (gaussienne)
x0_init = 0.25 #origine de la fonction d'onde psi centrÃ©e en x0
x0_cent = 0.5
k0_init = 300. #vecteur d'onde initial


#Valeurs liÃ©es au potentiel

V0_init = 100000. #Energie initiale du potentiel 
n1_init_puits = 50 #Position initiale du premier point correspondant Ã  une valeur non nulle du puits de potentiel harmonique
n1_init = 500  #Position initiale du premier point correpondant Ã  une valeur non nulle de la barriÃ¨re de potentiel
n2_init = 520  #Position initiale du dernier point correpondant Ã  une valeur non nulle de la barriÃ¨re de potentiel



"""
PREMIERE ETAPE : On construit l'hamiltonien H de notre systÃ¨me H = T + V
"""


"""
  On construit l'opÃ©rateur cinÃ©tique T
"""

Mat_1 = 2 * np.eye(n) # matrice diagonale avec des 2

v = - np.ones(n-1)
Mat_2 = np.diag(v,1) # matrice diagonale dÃ©calÃ© de un avec des -1
Mat_3 = np.diag(v,-1)  # matrice diagonale dÃ©calÃ© de un avec des -1

Mat_total = Mat_1 + Mat_2 + Mat_3 # on somme les trois matrices crÃ©Ã©es ---> Matrice tri-diagonale

T = np.zeros((n,n))
T = 1. / ( delta_x * delta_x ) * Mat_total # on crÃ©Ã© la matrice liÃ©e Ã  l'Ã©nergie cinÃ©tique (dÃ©rivÃ©e seconde dans l'espace)

"""
  On construit l'opÃ©rateur potentiel V
"""

"""
Nous utiliserons quatre potentiels diffÃ©rents : la barriÃ¨re de potentiel, la marche de potentiel, le potentiel nul et 
le puits harmonique.
Les fonctions utilisÃ©es ont pour variable la position initiale du potentiel, la position finale et sa valeur.
MÃªme si nous savons que les variables ne sont pas toutes prises en compte dans la fonction dÃ©finissant un potentiel, nous
gardons cette configuration pour ne pas crÃ©er de confusions lorsque nous dÃ©finirons l'algorithme de rÃ©solution (dans la suite). 
"""


def barriere_potentiel(n1,n2,V0):  # Matrice du potentiel V
    V = np.zeros((n,n))
    for i in range (n1,n2+1):
        V[i][i] = V0
    return(V)    

def trace_barriere(n1,n2,V0):  #TracÃ© du graphique caractÃ©risant notre potentiel V
    V_graph = np.zeros(n)
    for i in range(n1,n2+1):
        V_graph[i] = V0 * 0.00001
    return(V_graph)    

def potentiel_nul(n1,n2,V0):   # Matrice du potentiel V
    V = np.zeros((n,n))
    return(V)

def trace_nul(n1,n2,V0):    #TracÃ© du graphique caractÃ©risant notre potentiel V
    V_graph = np.zeros(n)
    return(V_graph)

def marche_potentiel(n1,n2,V0):   # Matrice du potentiel V
    V = np.zeros((n,n))
    for i in range(n1,n):
        V[i][i] = V0
    return(V)    
    
def trace_marche(n1,n2,V0):     #TracÃ© du graphique caractÃ©risant notre potentiel V
    V_graph = np.zeros(n)
    for i in range(n1,n):
        V_graph[i] = V0 * 0.00001
    return(V_graph)    

def puits_potentiel_harmonique(n1,n2,V0):  # Matrice du potentiel V
    V = np.zeros((n,n))
    # for i in range(n1_init_puits,n2+1):
    for i in range(n):
        V[i][i] = (X[i] - 0.5) * (X[i] - 0.5)*10*V0 #*1000000
    return(V)

def trace_puits_harmonique(n1,n2,V0):   #TracÃ© du graphique caractÃ©risant notre potentiel V
    V_graph = np.zeros(n)
    # for i in range (n1_init_puits,n2+1):
    for i in range(n):
        # V_graph[i] = (X[i] - 0.5) * (X[i]-0.5)* 5.
        V_graph[i] = V[i][i]*0.00001
    return(V_graph)    



def trace_energie(k0):     #TracÃ© du graphique caractÃ©risant l'Ã©nergie cinÃ©tique du paquet d'ondes (proportionnelle Ã  k0Â²)
    E_graph = np.ones(n) * k0 * k0 * 0.00001
    #$$$$ facteur /2 Ã  vÃ©rifier
    return(E_graph)
    


type_pot = "Barriere"
V_init = barriere_potentiel(n1_init,n2_init,V0_init) # barriÃ¨re de potentiel initiale construite


"""
  On conclut en construisant H
"""

H_init = T + V_init # Hamiltionien initial du systÃ¨me



"""
DEUXIEME ETAPE : On construit le paquet d'ondes gaussien psi
"""

def psi(x0,k0,sigma0):    # paramÃ¨tres: centrÃ© en x0 ; vecteur d'ondes k0 ; Ã©cart-type sigma0
    psi = np.zeros( n , dtype = np.complex )
    for k in range(1,n-1):
        x = delta_x * k
        psi[k] = ( np.cos(k0*x) + 1j * np.sin(k0*x) ) * ( np.exp( -(x-x0)*(x-x0) / (4.0 * sigma0 * sigma0) ) )
    return(psi)
    
def psi_carre(psi):       #fonction d'onde au carrÃ©                       
    return (np.real(psi)*np.conj(psi))

def module_psi(psi):      #module de la fonction d'onde
    return np.abs(psi)

def partie_reelle(psi):   #partie rÃ©elle de la fonction d'onde
    return np.real(psi)    

"""
TROISIEME ETAPE : On met en place un algorithme de rÃ©solution de l'Ã©quation de SchrÃ¶dinger dÃ©pendante du temps
                  pour dÃ©terminer les valeurs de psi Ã  chaque pas de temps.
                  ---> SchÃ©ma implicite unitaire
"""

"""
  On construit les deux opÃ©rateurs de l'Ã©quation de rÃ©solution.
"""

operateur_1 = np.eye(n) + 1j * (1./2.) * delta_t * H_init   # opÃ©rateur Ã  gauche de l'Ã©quation (initial)
operateur_2 = np.eye(n) - 1j * (1./2.) * delta_t * H_init   # opÃ©rateur Ã  droite de l'Ã©quation (initial)

#On dÃ©finit les vecteurs a,b,c de l'opÃ©rateur rÃ©duit 1 et les vecteurs d,e,f de l'opÃ©rateur rÃ©duit 2.
#Cela permet d'utiliser des vecteurs au lieu de matrices (de grandes tailles) afin de gagner de la mÃ©moire et du temps de calcul

b = np.zeros(n,dtype = np.complex)
e = np.zeros(n,dtype = np.complex)
for i in range(n):
    b[i] = operateur_1[i][i]
    e[i] = operateur_2[i][i]

a = np.zeros(n-1,dtype = np.complex)
d = np.zeros(n-1,dtype = np.complex)
for i in range(n-1):
    a[i] = operateur_1[i+1][i]
    d[i] = operateur_2[i+1][i]

c = np.zeros(n-1,dtype = np.complex)
f = np.zeros(n-1,dtype = np.complex)
for i in range(n-1):
    c[i] = operateur_1[i][i+1]
    f[i] = operateur_2[i][i+1]


"""
On rÃ©alise un algorithme de rÃ©solution AX=Y oÃ¹ A est une matrice tridiagonale de diagonales a,b,c et Y=d, basÃ© sur l'algorithme
de Thomas.
"""

def TDMASolve(a, b, c, d):
    n = len(d)
    res = np.zeros(n,dtype = np.complex)
    c_prime = np.zeros(n-1, dtype = np.complex)
    d_prime = np.zeros(n, dtype = np.complex)
    c_prime[0] = c[0]/b[0]
    d_prime[0] = d[0]/b[0]
    # for i in range(1,n-1):
    #     c_prime[i] = c[i] / (b[i] - a[i]*c_prime[i-1])    
    # for j in range(1,n-1):
    #     d_prime[j] = (d[j] - a[j] * d_prime[j-1]) / (b[j] - a[j] * c_prime[j-1])    
    for i in range(1,n-1):
        c_prime[i] = c[i] / (b[i] - a[i-1]*c_prime[i-1])    
    for j in range(1,n):
        d_prime[j] = (d[j] - a[j-1] * d_prime[j-1]) / (b[j] - a[j-1] * c_prime[j-1])    
    #final
    res[n-1] = d_prime[n-1]
    for i in range(n-2,-1,-1):
        res[i] = (d_prime[i] - c_prime[i]*res[i+1])
    return(res)     
        
    
"""
On crÃ©Ã© une fonction produit matriciel permettant de faire le produit entre une matrice tridiagonale A composÃ©e de trois diagonales
d ( dim (n-1)*1), e ( dim n*1), f ( dim (n-1)*1) et d'un vecteur B de dimension (n*1)
"""

def produit_matriciel_special(d,e,f,B):
    n = len(B)
    res = np.zeros(n,dtype = np.complex)
    #initialisation
    res[0] = e[0] * B[0] + f[0] * B[1]
    #algorithme produit
    for i in range(1,n-1):
        # res[i] = d[i] * B[i-1] + e[i] * B[i] + f[i] * B[i+1]
        res[i] = d[i-1] * B[i-1] + e[i] * B[i] + f[i] * B[i+1]
    #final
    res[n-1] = d[n-2] * B[n-2] + e[n-1] * B[n-1]
    return(res)




"""
QUATRIEME ETAPE : RÃ©solution et tracÃ© du paquet d'ondes
"""


"""
Initialisation
"""

#On dÃ©finit l'abscisse des x

X = np.zeros(n)    # espace des x
for i in range(n):
    X[i] = i * delta_x
       

#Valeurs/Conditions initiales

psi_0 = psi(x0_init,k0_init,sigma_init)
k0_new = k0_init
sigma_new=sigma_init
x0_new = x0_init
V0_new=V0_init
n2_new=n2_init
V = V_init

fig = plt.figure() #figure animation
ax=plt.axes(xlim=(0,1.),ylim=(-1.2,2.05))  #Axes x et y
plt.subplots_adjust(left = 0.3,bottom=0.45)  #on place le graphique sur la page
courbe, =ax.plot(X,partie_reelle(psi_0)) #tracÃ© initiale de la partie rÃ©elle du paquet d'ondes
courbe1, =ax.plot(X,module_psi(psi_0)) #tracÃ© initiale du module du paquet d'ondes

p, = ax.plot(X,trace_barriere(n1_init,n2_init,V0_init),linewidth = 1) #tracÃ© initial du potentiel
z, = ax.plot(X,trace_energie(k0_init),'--',linewidth = 1) #tracÃ© initial de l'Ã©nergie cinÃ©tique (en pointillÃ©)

"""
On dÃ©finit dans la partie suivante des curseurs et des boutons qui permettront Ã  l'utilisateur du programme de pouvoir
modifier certaines variables du systÃ¨me (liÃ©es Ã  la fonction d'onde ou au potentiel)
"""

axSlider1 = plt.axes([0.1,0.25,0.8,0.05])             #Slider 1 >>> vecteur d'onde k0
k0Slider = Slider(axSlider1,'k0',1.,1000.,k0_init)

axSlider2 = plt.axes([0.1,0.2,0.8,0.05])              #Slider 2 >>> sigma
sigmaSlider = Slider(axSlider2,'sigma',0.001,0.15,sigma_init)  

axSlider3 = plt.axes([0.1,0.15,0.8,0.05])             #Slider 3 >>> x0
x0Slider = Slider(axSlider3,'x0',0.15,0.5,x0_init)

axSlider4 = plt.axes([0.1,0.06,0.8,0.05])             #Slider 4 >>> largeur de la barriÃ¨re
n2Slider = Slider(axSlider4,'largeur',n1_init,n1_init+4*(n2_init-n1_init),n2_init)

axSlider5 = plt.axes([0.1,0.01,0.8,0.05])             #Slider 2 >>> Ã©nergie du potentiel V0
V0Slider = Slider(axSlider5,'V0',0,200000,V0_init)

rax = plt.axes([0.05, 0.7, 0.15, 0.15])               #Boutons radio >>> modification de la forme du potentiel
radio = RadioButtons(rax, ('Barriere', 'Marche', 'Nul','Puits'))
       

def recalc_H():
    global psi_0,V,a,b,c,d,e,f
    
    H = T + V # Hamiltonien modifiÃ©   
    operateur_1 = np.eye(n) + 1j * (1./2.) * delta_t * H   # opÃ©rateur Ã  gauche de l'Ã©quation modifiÃ© 
    operateur_2 = np.eye(n) - 1j * (1./2.) * delta_t * H   # opÃ©rateur Ã  droite de l'Ã©quation modifiÃ©
    
    b = np.zeros(n,dtype = np.complex)
    e = np.zeros(n,dtype = np.complex)
    for i in range(n):
        b[i] = operateur_1[i][i]
        e[i] = operateur_2[i][i]
    a = np.zeros(n-1,dtype = np.complex)
    d = np.zeros(n-1,dtype = np.complex)
    for i in range(n-1):                     #opÃ©rateurs rÃ©duits modifiÃ©s
        a[i] = operateur_1[i+1][i]
        d[i] = operateur_2[i+1][i]
    c = np.zeros(n-1,dtype = np.complex)
    f = np.zeros(n-1,dtype = np.complex)
    for i in range(n-1):
        c[i] = operateur_1[i][i+1]
        f[i] = operateur_2[i][i+1]
    
    psi_0 = psi(x0_new,k0_new,sigma_new)    #Nouvelle fonction d'onde
    z.set_ydata(trace_energie(k0_new))      #TracÃ© de la nouvelle Ã©nergie cinÃ©tique

def update(val):    
    global x0_new, k0_new,sigma_new
    sigma_new = sigmaSlider.val
    k0_new = k0Slider.val
    x0_new = x0Slider.val            #Les variables sont modifiÃ©es quand on les choisit avec les curseurs
    # on recalcul H
    recalc_H()

def update1(val):    
    global n2_new, V0_new
    n2_new = int(n2Slider.val)
    V0_new = V0Slider.val
    # on recalcul V
    potfunc(type_pot)
  

sigmaSlider.on_changed(update)
k0Slider.on_changed(update)
x0Slider.on_changed(update)
n2Slider.on_changed(update1)
V0Slider.on_changed(update1)

def potfunc(label):      #AprÃ¨s avoir cliquÃ© sur le bouton correspondant, le potentiel est modifiÃ© ainsi que le tracÃ©
    global V,psi_0,H,a,b,c,d,e,f,type_pot,x0_new
    type_pot = label
    if label == 'Barriere' :
        V = barriere_potentiel(n1_init,n2_new,V0_new)
        p.set_ydata(trace_barriere(n1_init,n2_new,V0_new))
        x0_new = x0_init
    if label == 'Marche' :
        V = marche_potentiel(n1_init,n2_new,V0_new)
        p.set_ydata(trace_marche(n1_init,n2_new,V0_new))
        x0_new = x0_init
    if label == 'Nul' :
        V = potentiel_nul(n1_init,n2_new,V0_new)
        p.set_ydata(trace_nul(n1_init,n2_new,V0_new))
        x0_new = x0_init
    if label == 'Puits':
        V = puits_potentiel_harmonique(n1_init,n2_new,V0_new)
        p.set_ydata(trace_puits_harmonique(n1_init,n2_new,V0_new))
        x0_new = x0_cent
    x0Slider.set_val(x0_new)
    
    #On met Ã  jour les opÃ©rateurs    
    
    H = T + V    
    operateur_1 = np.eye(n) + 1j * (1./2.) * delta_t * H   # opÃ©rateur Ã  gauche de l'Ã©quation
    operateur_2 = np.eye(n) - 1j * (1./2.) * delta_t * H   # opÃ©rateur Ã  droite de l'Ã©quation
    
    b = np.zeros(n,dtype = np.complex)
    e = np.zeros(n,dtype = np.complex)
    for i in range(n):
        b[i] = operateur_1[i][i]
        e[i] = operateur_2[i][i]

    a = np.zeros(n-1,dtype = np.complex)
    d = np.zeros(n-1,dtype = np.complex)
    for i in range(n-1):
        a[i] = operateur_1[i+1][i]
        d[i] = operateur_2[i+1][i]

    c = np.zeros(n-1,dtype = np.complex)
    f = np.zeros(n-1,dtype = np.complex)
    for i in range(n-1):
        c[i] = operateur_1[i][i+1]
        f[i] = operateur_2[i][i+1]    
    
    psi_0 = psi(x0_new,k0_new,sigma_new) #fonction d'onde mise Ã  jour

radio.on_clicked(potfunc) 

"""
fonction animate : C'est elle qui permet l'animation du paquet d'ondes dans le temps.
                   On rÃ©sout par mÃ©thode rÃ©cursive l'Ã©quation de SchrÃ¶dinger : psi_1 est la mise Ã  jour de psi_0 aprÃ¨s
                   un pas de temps. 
"""

def animate(i):  
    global psi_0    
    #print(i)
    for k in range(10): #On affiche la fonction d'onde aprÃ¨s 10 pas de temps
        #rÃ©solution        
        terme_de_droite = produit_matriciel_special(d,e,f,psi_0)
        psi_1 = TDMASolve(a,b,c,terme_de_droite)
        psi_0 = psi_1
    courbe.set_ydata(partie_reelle(psi_1)) #On trace la partie rÃ©elle de la fonction d'onde
    courbe1.set_ydata(module_psi(psi_1)) #On trace le module de la fonction d'onde
    return courbe,  

ani = anim.FuncAnimation(fig, animate, 400) #pour lancer l'animation
 
plt.show()


"""
DÃ©fauts:
Pour mettre Ã  jour les caractÃ©ristiques du paquet d'ondes et du potentiel aprÃ¨s avoir utiliser les curseurs, il faut 
rÃ©appuyer sur le bouton Radio correspondant au potentiel souhaitÃ©.
"""