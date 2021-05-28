import math
import numpy as np
import scipy as sc
from scipy import sparse 
from scipy import integrate 
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from IPython.display import HTML
from numba import jit, njit


''' This project is mostly a recreation of a blog post by Cory Chu at
https://blog.gwlab.page/solving-1-d-schrodinger-equation-in-python-dcb3518ce454. 
After following some of his work, I also attempted to do his suggested simulations.
The goal of this project was to solve the Schrödinger Equation for a SHO. After this
I will make an animation of some osscilations in the model.  '''

#Start by defining our conditions
dx = 0.02 #Spatial seperation
x = np.arange(0,10,dx) #Create our spatial dimension


#Wave Packet conditions
k = 0.1 #Wave number 
m = 1 #mass 
sigma = 0.1 #Width of Gaussian
x0 = 3 #Center of initial packet


Norm = 1 / (sigma * np.sqrt(sc.constants.pi)) #Normalization Constant


#Initial Wavefunction for Wavepacket
psi0 = np.sqrt(Norm)* np.exp(-(x-x0)/(2*sigma)) *np.exp(1j*k*x)

#Creating our potential 
Vxmin = 5 #Minimum point for potential
T = 1  #Period of SHO
w = (2 * np.pi)/ T#Angular Velocity
scaling = 0.001 #Scaling factor 
V = ((k/2)*(x - Vxmin)**2)*scaling #Potential function


#Solve the intial conditions for our SE
dt = 0.005 #Time spacing
t0 = 0 #Initial Time    
tf = 1 #Final Time
t_interval = np.arange(t0,tf,dt)

#We then create our laplace operator which will convert the wavefunction into a single variable function.
#The matrix which is our laplace operator is banded, so we can use scipy's sparse feature.
D2 = sc.sparse.diags([1,-2,1],
                    [-1,0,1],
                    shape=(x.size, x.size)) / dx**2

#Defining our Schrodinger Equation as psi_t
def psi_t(t,psi):
    -1j*((sc.constants.hbar/(2*m))*D2.dot(psi) + (1/sc.constants.hbar)*V*psi)


IVPSol = sc.integrate.solve_ivp(psi_t,t_span=[t0,tf], y0 = psi0, t_eval = t_interval, method = 'RK23')


