#%% 
import numpy as np
import matplotlib.pyplot as plt
#%%
lambda_ = 550E-9 # m
ve = 30E3 #m/s
c = 3E8 # m/s
L1 = 11 #m
L2 = 11 #m


# beta = ve/c
# phi = 2*np.pi*(2*(L2-L1)+(2*L2-L1)*beta**2)/lambda_
# phip = 2*np.pi*(2*(L2-L1)+(L2-2*L1)*beta**2)/lambda_
# Phi = phip-phi
# print(Phi/2/np.pi)

x = np.arange(0, 10, 0.01)

def Phi(ve, L1, L2, lambda_): 
    beta = ve/c
    phi = 2*np.pi*(2*(L2-L1)+(2*L2-L1)*beta**2)/lambda_
    phip = 2*np.pi*(2*(L2-L1)+(L2-2*L1)*beta**2)/lambda_
    Phi = phip-phi
    return Phi

def michelson_morley(x,ve, L1, L2, lambda_):
    I0 = 1
    I = 2*I0*(1+np.cos(x+Phi(ve, L1, L2, lambda_)))
    return I


plt.plot(x,michelson_morley(x,ve, L1, L2, lambda_))
plt.plot(x,michelson_morley(x,c, L1, L2, lambda_))

PhiMM = Phi(ve, L1, L2, lambda_)
PhiMMes = PhiMM/2/np.pi
print("Résultat de Michelson et Morley Phi/2pi = %.3f" % PhiMMes)

#%%
