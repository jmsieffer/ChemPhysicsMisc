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

from timeit import default_timer as timer

start = timer()


''' NOT COMPLETED. This project is mostly a recreation of a blog post by Cory Chu at
https://blog.gwlab.page/solving-1-d-schrodinger-equation-in-python-dcb3518ce454. 
After following some of his work, I also attempted to do his suggested simulations.
The goal of this project was to solve the Schrödinger Equation for a SHO. After this
I will make an animation of some osscilations in the model. Eventually abondoned. '''

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
psi0 = np.sqrt(Norm)* np.exp(-(x-x0)**2/(2*sigma**2)) *np.exp(1j*k*x)

#Creating our potential 
Vxmin = 5 #Minimum point for potential
T = 1  #Period of SHO
#w = (2 * np.pi)/ T#Angular Velocity
 
V = ((k/2)*(x - Vxmin)**2) #Potential function
#V = 0.5 * k * (x - Vxmin)**2

plt.plot(x, V, "k--", label=r"$V(x) = \frac{1}{2}m\omega^2 (x-5)^2$ (x0.01)")
plt.plot(x, np.abs(psi0)**2, "r", label=r"$\vert\psi(t=0,x)\vert^2$")
plt.legend(loc=1, fontsize=8, fancybox=False)
print("Total Probability: ", np.sum(np.abs(psi0)**2)*dx)

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
    return -1j*((sc.constants.hbar/(2*m))*D2.dot(psi) + (1/sc.constants.hbar)*V*psi)

test_psi = psi_t(5, psi0)
plt.plot(x,test_psi)

IVPSol = sc.integrate.solve_ivp(psi_t, t_span = [t0, tf], y0 = psi0, t_eval = t_interval, method="RK45")


fig = plt.figure(figsize=(6, 4))
for i, t in enumerate(IVPSol.t):
    plt.plot(x, np.abs(IVPSol.y[:,i])**2)             # Plot Wavefunctions
    print(np.sum(np.abs(IVPSol.y[:,i])**2)*dx)        # Print Total Probability (Should = 1)
plt.plot(x, V)   # Plot Potential
plt.legend(loc=1, fontsize=8, fancybox=False)

fig = plt.figure()
ax1 = plt.subplot(1,1,1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)
title = ax1.set_title('')
line1, = ax1.plot([], [], "k--")
line2, = ax1.plot([], [])


def init():
    line1.set_data(x, V * 0.01)
    return line1,
def animate(i):
    line2.set_data(x, np.abs(IVPSol.y[:,i])**2)
    title.set_text('Time = {0:1.3f}'.format(IVPSol.t[i]))
    return line1,
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(IVPSol.t), interval=50,                 
                               blit=True)

anim.save('sho.mp4', fps=15, dpi=600)

# Display the animation in the jupyter notebook
HTML(anim.to_jshtml())

elapsed_time = timer() - start
