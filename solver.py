#here the equations will be discretized
import numpy as np
import matplotlib.pyplot as plt
from velocity import velocity,diffusivity,grid

#Define functions to discretize the main equations
#Upwind discretization for advection terms

def advection_upwind (C,U,V,dx,dy):
    Nx = 100
    Ny = 100
    #Nx,Ny = C.shape #define grid size, gets it from the size of the input C1/C2
    dCdx = np.zeros_like(C) #derivative in x
    dCdy = np.zeros_like(C) #derivative in y
    #start discretization, go inside the matrix
    for i in range(1, Nx-2): #goes inside the x coordinate
        for j in range(1, Ny-2): #goes inside the y coordinate
            # Upwind differencing for x-direction
            if U[i, j] > 0: #case that U is >0 --> backward difference
                dCdx[i, j] = (C[i, j] - C[i - 1, j]) / dx
            else:   #case that U is <=0 --> forward difference
                dCdx[i, j] = (C[i + 1, j] - C[i, j]) / dx

            # Upwind differencing for y-direction
            if V[i, j] > 0: #case that V is >0 --> backward difference
                dCdy[i, j] = (C[i, j] - C[i, j - 1]) / dy #I think the mistake is from how I defined dy
            else: #case that V is <=0 --> forward difference
                dCdy[i, j] = (C[i, j + 1] - C[i, j]) / dy

    return dCdx, dCdy

def harmonic_avg(a, b): #Defines harmonic average
    return 2 * a * b / (a + b + 1e-12)  # definition of harmonic average, plus it avoids a division by zero by adding the +1e-12

def diffusion_flux(C, K, dx, dy):
    Nx,Ny = C.shape
    diff_flux = np.zeros_like(C) #flux of diffusion, same size as C

    for i in range(1, Nx-2): #enter the matrix
        for j in range(1, Ny-2): #enter the matrix
            # Diffusion in x-direction
            Ke = harmonic_avg(K[i, j], K[i + 1, j]) #calculates the harmonic average of the diffusivity east
            Kw = harmonic_avg(K[i, j], K[i - 1, j]) #calculates the harmonic average of the diffusivity west
            diff_flux[i, j] += (Ke * (C[i + 1, j] - C[i, j]) - Kw * (C[i, j] - C[i - 1, j])) / dx ** 2 #Calculcates the diffusive flux in x

            # Diffusion in y-direction
            Kn = harmonic_avg(K[i, j], K[i, j + 1]) #calculates the harmonic average of the diffusivity north
            Ks = harmonic_avg(K[i, j], K[i, j - 1]) #calculates the harmonic average of the diffusivity south
            diff_flux[i, j] += (Kn * (C[i, j + 1] - C[i, j]) - Ks * (C[i, j] - C[i, j - 1])) / dy ** 2 #Calculcates the diffusive flux in x

    return diff_flux

#Create Grid
#define grid size
Nx= 100
Ny = 100
X,Y,x,y = grid(Nx, Ny)
dx = 1.0 / (Nx - 1)
dy = (0.5 - (-0.5)) / (Ny - 1)
#Getting U,V and K values
U,V,e = velocity(X,Y)
K = diffusivity(X,Y)

#initialize C1 and C2 fields
C1 = np.zeros((Nx, Ny))  # Correct
C2 = np.zeros((Nx, Ny))  # Correct


#Euler foward to calculate the time derivative
dt = 0.01
timestep = 100
Ar = 0
#Ar = 20 #if reaction is happening
# Time loop
for t in range(timestep):
    # 1. Compute fluxes (advection, diffusion, and reaction)
    adv_flux_C1 = advection_upwind(C1, U, V, dx, dy) #Calculate advection C1
    adv_flux_C2 = advection_upwind(C2, U, V,dx,dy) #Calculate advection C2

    diff_flux_C1 = diffusion_flux(C1, K,dx, dy) #calculate diffusion C1
    diff_flux_C2 = diffusion_flux(C2, K, dx, dy) #calculate diffusion C2

    reaction_C1 = Ar * C1 * C2 #calculate the reaction (relevant only if Ar=!0)
    reaction_C2 = Ar * C1 * C2 #calculate the reaction

    # Euler forward update
    C1_new = C1 + dt * (adv_flux_C1 + diff_flux_C1 + reaction_C1) #update C1
    C2_new = C2 + dt * (adv_flux_C2 + diff_flux_C2 + reaction_C2) #update C2

    # Apply boundary conditions
    C2_new[0, :] = 0  # West boundary for C2
    C1_new[0,:] = 1 # West boundary for C1, missing!!
    C1[:, -1] = C1[:, -2]  # East BC for C1 (zero gradient?)
    C2[:, -1] = C2[:, -2]  # East BC for C2
    C2_new[:, 0] = 0  # South boundary for C2
    C1_new[:, 0] = 0  # South boundary for C1
    C2_new[:, -1] = 0  # North boundary for C2
    C1_new[:, -1] = 0  # North boundary for C1

    #Update C1 and C2
    C1 = C1_new
    C2 = C2_new
