import matplotlib.pyplot as plt
import numpy as py

#defining the grid for the problem 
def grid(Nx = 100, Ny = 100, x_range = [0,1], y_range = [-0.5,0.5]):
    dx = (x_range[1]-x_range[0])/(Nx-2)
    dy = (y_range[1]-y_range[0])/(Ny-2)
    x = py.linspace(x_range[0]-dx/2, x_range[1]+dx/2, Nx)#Discretizing the x_range into 100 bits
    y = py.linspace(y_range[0]-dy/2, y_range[1]+dy/2, Ny)#Discretizing the y_range into 100 bits
    X,Y = py.meshgrid(x,y, indexing = "ij")#X and Y are 2D arrays representing the grid
    return X,Y,x,y

#Calculating Velocity 
def velocity(X,Y):
    #constants in calculating velocities
    VO = -0.5#Virtual Origin 
    s = 7.67#value of sigma (given)
    Uin = 1 #Value of inlet velocity
    SH = 0.01 #Value of slit head at inlet
    Kbig = SH*Uin**2#Given value:Defining K
    ymp = 0#Given Value

    #Given supporting equations 
    xVO = X + VO#using definition of xVO
    e = s*(Y-ymp)/(xVO)#using definition of eta

    #Using equations of velocities in x and y direction (U and V respectively)

    U = 0.5*py.sqrt(py.clip((3*Kbig*s)/(xVO), 0, None)) * (1-py.tanh(e)**2)#Definition of velocity in x direction (West to east)
    V = 0.25*py.sqrt(py.clip((3*Kbig)/(s*xVO), 0, None))*(2*e*(1-py.tanh(e)**2)-py.tanh(e))#Definition of velocity in y direction

    return U,V,e #Returning the value of U and V

#Calculating extra Diffusivity and total diffusivity
def diffusivity(X,Y):
    e_half = 0.88137#Given value of eta half
    VO = -0.5#Virtual 
    xVO = X + VO#using definition of xVO
    s = 7.67#value of sigma (given)

    b_half = e_half*(xVO)/s#definition of bhalf given
    U, V, e = velocity(X,Y)#calling velocity U from velocity function
    Ucl = U/(0.5*(py.clip(1-py.tanh(e)**2,0, None)))#using definition of Ucl = U/f(eta)
    eps_T = 0.037*b_half*Ucl#Given definition of epsilon
    K_t = 2*eps_T
    K_p = 1e-3#Given Value of physical diffusivity in m^2/s

    #applying condition for k
    K = py.where(e< 3 * e_half, K_p + K_t, K_p)
    return K

#Create Grid
X,Y,x,y = grid(Nx = 100, Ny = 100)

#Getting U,V and K values 
U,V,e = velocity(X,Y)
K = diffusivity(X,Y)

#Plotting U 
fig = plt.figure(figsize = (12,6))
ax = fig.add_subplot(121, projection = '3d')
ax.plot_surface(X,Y,U, cmap = 'viridis')
ax.set_title('Velocity (U) in x-direction')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Velocity (U)')
plt.show()




