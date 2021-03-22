#%%
import numpy as np
import scipy as sp
from scipy import constants as sc
from scipy import integrate as integrate
import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.axis as ax 
import pandas as pd

#Simulating Einstein Model of Heat Capacity for Aluminum

#Constants
Vs = 6320 
n = 6.03*(10**28)
ET = 290
DT = 428 
omega = (6 * np.pi**2 * n * Vs**3)**(1/3)

#Temperature Input
T = np.linspace(0,1000,100)
Tplot = T/ET
x = (sc.hbar * omega) / (sc.k * T)

#Einstein Function
def CEin(T):
    return 3 * sc.N_A * sc.k * ((ET / T )**2) * ((np.exp((sc.hbar * omega) / (sc.k * T))) / ((np.exp((sc.hbar * omega) / (sc.k * T))-1)**2))

CE = CEin(T) 

#Plotting the result
fig = plt.subplots()
matplotlib.rcParams['font.serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "serif"
plt.ylabel('Heat Capacity ', fontsize = '12')
plt.xlabel('$\Theta_E / T$', fontsize = '12')
plt.ylim(0, 4)
plt.xlim(0,2.34)
plt.title('Einstein Heat Capacity for Al')
plt.annotate(r'$C = 3Nk_b(\frac{\Theta_E}{T})^2\frac{e^{\frac{\hbar\omega}{k_b T}}}{(e^{\frac{\hbar\omega}{k_b T}}-1)^2}$', xy=(0.05, 0.90), xycoords='axes fraction')

plt.plot(Tplot,CE, color= 'black')


# %%
