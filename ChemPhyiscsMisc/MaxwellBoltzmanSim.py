#%%
import numpy as np
import scipy as sp
from scipy import constants as sc
from scipy import integrate as integrate
import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.axis as ax 
import pandas as pd

#Simulating Maxwell Boltzmann Distribution of  elemental Gasses. All mean velocities calculated as well. ;All simulations run at at 298 K, but this can be modified.
#Creating a velocity variable
V = np.linspace(0,6000,1000)

#Putting all of the Gas Data in one DataFrame
Elements = {'Element': ['H', 'N3', 'O2', 'O3','F2' ,'Cl2', 'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'],
        'Atomic Mass (AU)': [1.008, 42.021, 32, 48, 37.996, 70.9, 4.0026, 20.18, 39.948, 83.798, 131.29, 222]}
Edf = pd.DataFrame(Elements, columns = ['Element', 'Atomic Mass (AU)'])
Edf['Mass (KG)'] = (Edf['Atomic Mass (AU)'] / sc.N_A) / 1000


#3D Maxwell-Boltzman Distribution
def ThDMB(m, T, V):
    return 4 * np.pi * V**2 * (m/(2 * np.pi * sc.k * T))**(3/2) * np.exp((-m * V**2)/(2 * sc.k * T))

def VMP(m, T):
    return np.sqrt((2 * sc.k * T)/m)

def VM(m, T):
    return np.sqrt((8 * sc.k * T)/(np.pi * m))

def VRMS(m, T):
    return np.sqrt((3 * sc.k * T)/m)


#Creating a list of all outputs
Odf = tuple([ThDMB(x, 298, V) for x in Edf['Mass (KG)']])

#Adding Velcotities to the DataFrame
Edf['Most Probable Velocity (m/s)'] = [VMP(x, 298) for x in Edf['Mass (KG)']]
Edf['Mean Velocity (m/s)'] = [VM(x, 298) for x in Edf['Mass (KG)']]
Edf['RMS Velocity (m/s)'] = [VRMS(x, 298) for x in Edf['Mass (KG)']]



#Plotting the all gasses toegether
fig = plt.subplots()
matplotlib.rcParams['font.serif'] = "Palatino"
matplotlib.rcParams['font.family'] = "serif"
plt.ylabel(r'$\rho (x)$', fontsize = '12')
plt.xlabel(r'$\nu (m/s)$', fontsize = '12')
plt.ylim(0, .006)
plt.xlim(0,6000)
plt.title('Maxwell-Boltzman Distribution for Elemental Gasses')

plt.plot(V,Odf[0], color= 'red', label = Edf['Element'][0])
plt.plot(V,Odf[1], color= 'darkorange', label = Edf['Element'][1])
plt.plot(V,Odf[2], color= 'sandybrown', label = Edf['Element'][2])
plt.plot(V,Odf[3], color= 'yellow', label = Edf['Element'][3])
plt.plot(V,Odf[4], color= 'forestgreen', label = Edf['Element'][4])
plt.plot(V,Odf[5], color= 'darkgreen', label = Edf['Element'][5])
plt.plot(V,Odf[6], color= 'teal', label = Edf['Element'][6])
plt.plot(V,Odf[7], color= 'lightblue', label = Edf['Element'][7])
plt.plot(V,Odf[8], color= 'darkslateblue', label = Edf['Element'][8])
plt.plot(V,Odf[9], color= 'indigo', label = Edf['Element'][9])
plt.plot(V,Odf[10], color= 'darkblue', label = Edf['Element'][10])
plt.plot(V,Odf[11], color= 'black', label = Edf['Element'][11])
plt.legend(loc="upper right")

plt.plot(V, Odf[0], color = 'black')
# %%
