#%%
import numpy as np
import scipy as sp
from scipy import constants as sc
import matplotlib
import matplotlib.pyplot as plt  
import pandas as pd

'''Simulating Maxwell Boltzmann Distribution of  elemental Gasses. All mean velocities 
calculated as well. This script outputs 2 (or however many desired) scalings of the 
MB distribution for all elemental gasses in one graph. It also outputs one graph
for each gas individually and has labeled the different locations of the velocities
of interest.  All simulations run at at 298 K, but this can be modified. '''

#Creating a velocity variable
V = np.linspace(0,6000,1000)

# Putting all of the Gas Data in one DataFrame
Elements = {'Element': ['H2', 'N3', 'O2', 'O3','F2' ,'Cl2', 'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'],
        'Atomic Mass (AU)': [2.016, 42.021, 32, 48, 37.996, 70.9, 4.0026, 20.18, 39.948, 83.798, 131.29, 222]}
Edf = pd.DataFrame(Elements, columns = ['Element', 'Atomic Mass (AU)'])
Edf.insert(loc = 2, column = "Titles", value = ['Maxwell-Boltzman Distribution for Hydrogen Gas', 'Maxwell-Boltzman Distribution for Nitrogen Gas',
                                                'Maxwell-Boltzman Distribution for Oxygen Gas', 'Maxwell-Boltzman Distribution for Ozone',
                                                'Maxwell-Boltzman Distribution for Fluorine Gas', 'Maxwell-Boltzman Distribution for Chlorine Gas', 
                                                'Maxwell-Boltzman Distribution for Helium', 'Maxwell-Boltzman Distribution for Neon', 
                                                'Maxwell-Boltzman Distribution for Argon', 'Maxwell-Boltzman Distribution for Krypton',
                                                'Maxwell-Boltzman Distribution for Xenon', 'Maxwell-Boltzman Distribution for Radon',])
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


# Creating a list of all outputs
Odf = tuple([ThDMB(x, 298, V) for x in Edf['Mass (KG)']])

# Adding Velcotities to the DataFrame
Edf['Most Probable Velocity (m/s)'] = [VMP(x, 298) for x in Edf['Mass (KG)']]
Edf['Mean Velocity (m/s)'] = [VM(x, 298) for x in Edf['Mass (KG)']]
Edf['RMS Velocity (m/s)'] = [VRMS(x, 298) for x in Edf['Mass (KG)']]

# These are the maximum velocities for the two graphs. Each are good for displaying different
# properties. Fewer or more can be added to this list to create fewer or more graphs. 
svl = [1000, 6000]

#Plotting the all gasses toegether
for x in range(len(svl)):
    fig = plt.subplots()

    #Set fonts
    matplotlib.rcParams['font.serif'] = "Palatino"
    matplotlib.rcParams['font.family'] = "serif"

    #Set titles and axis labels
    plt.ylabel(r'$\rho (x)$', fontsize = '12')
    plt.xlabel(r'$\nu (m/s)$', fontsize = '12')
    plt.ylim(0, .006)
    plt.xlim(0, svl[x])
    plt.title('Maxwell-Boltzman Distribution for Elemental Gasses')

    #Plot each compound of interest
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

    #Legend
    plt.legend(loc="upper right")

#Creating an individual plot for each gas
for x in range(len(Odf)):
    fig = plt.subplots()
    # This defines all of the speeds that will show up on the plot.
    PVMP = ThDMB(Edf['Mass (KG)'][x], 298, Edf['Most Probable Velocity (m/s)'][x])
    PVM = ThDMB(Edf['Mass (KG)'][x], 298, Edf['Mean Velocity (m/s)'][x])
    PVRMS = ThDMB(Edf['Mass (KG)'][x], 298, Edf['RMS Velocity (m/s)'][x])

    #Font selection
    matplotlib.rcParams['font.serif'] = "Palatino"
    matplotlib.rcParams['font.family'] = "serif"

    # This is algorithm finds the velocity at the maximum probability of the distribution and scales it
    # by some constant such that the functions shape for each gas is nearly identical. The baseline for
    # each graph will be different as a result. For consistent baselines, simply change the xlim to the
    # desired maximum velocity value. 
    xlen = V[np.argmax(Odf[x])] * 3.2
    plt.xlim(0,xlen)

    # This algorithm does the same as above but for the y axis. Again this can be scaled as desired.
    plt.ylim(0,(Odf[x].max())*1.1)

    # Axis labels and Title
    plt.ylabel(r'$\rho (x)$', fontsize = '12')
    plt.xlabel(r'$\nu (m/s)$', fontsize = '12')
    plt.title(Edf['Titles'][x])

    # Plotting the function and the three speeds of interest. 
    plt.plot(V,Odf[x], color = 'black')
    plt.plot(Edf['Most Probable Velocity (m/s)'][x], PVMP, marker='o', markersize=5, color="grey", label = 'Most Probable Velocity')
    plt.plot(Edf['Mean Velocity (m/s)'][x], PVM, marker='s', markersize=5, color="grey", label = 'Mean Velocity')
    plt.plot(Edf['RMS Velocity (m/s)'][x], PVRMS, marker='^', markersize=5, color="grey", label = 'RMS Velocity')

    #Legend
    plt.legend(loc="upper right")

# %%
