#here the equations will be discretized
import numpy as np
from velocity import velocity,diffusivity,grid

#Define functions to discretize the main equations

#Upwind discretization for advection terms
def advection_upwind (C,U,V,dx,dy):
    Nx,Ny = C.shape #define grid size, gets it from the size of the input C1/C2
    dCdx = np.zeros_like(C) #derivative in x
    dCdy = np.zeros_like(C) #derivative in y
    #start discretization, go inside the matrix
    for i in range(1, Nx - 1): #goes inside the x coordinate
        for j in range(1, Ny - 1): #goes inside the y coordinate
            # Upwind differencing for x-direction
            if U[i, j] > 0: #case that U is >0 --> backward difference
                dCdx[i, j] = (C[i, j] - C[i - 1, j]) / dx
            else:   #case that U is <=0 --> forward difference
                dCdx[i, j] = (C[i + 1, j] - C[i, j]) / dx

            # Upwind differencing for y-direction
            if V[i, j] > 0: #case that V is >0 --> backward difference
                dCdy[i, j] = (C[i, j] - C[i, j - 1]) / dy
            else: #case that V is <=0 --> forward difference
                dCdy[i, j] = (C[i, j + 1] - C[i, j]) / dy

#Create Grid
#define grid size
Nx,Ny = 100
X,Y,x,y = grid(Nx, Ny)
#Getting U,V and K values
U,V,e = velocity(X,Y)
K = diffusivity(X,Y)

#initialize C1 and C2 fields
C1 = np.zeros(Nx,Ny)
C2 = np.zeros (Nx,Ny)

