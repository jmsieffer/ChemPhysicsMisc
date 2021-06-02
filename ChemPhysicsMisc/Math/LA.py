#%%
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt 

A = np.array([[1, 1, 0], [1, 2, 2], [0, 2, -1]])
B = np.array([[1, -1, 1], [-1, 0, 0], [1, 0, 1]])

def Com(A,B):
    ABProd = A.dot(B)
    BAProd = B.dot(A)
    C = np.subtract(ABProd, BAProd)
    return C
    
CAB = Com(A,B)


# %%
