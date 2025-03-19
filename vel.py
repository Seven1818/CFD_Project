from math import *
import matplotlib.pyplot as plt
import numpy as np

ftseries = open("timeseries.txt", "a")
ftseries.write("time fluxtot Tmax  \n")


timeswidtheps = 3 # limit turbulent diffusivity to 3 half widths
# started from diffusion_2D_inst.py
imax = 32
jmax = 32

Ubig = 1.
# parameters for obstacle wake
diameter = .01 # diameter of the obstacle creating the wake
VO       = 0.5 # distance of obstacle to the left of x = 0
ymp      = 0.0 # centerpoint obstacle
# free 2D jet
Kbig = diameter*Ubig**2
print("Kbig = ", Kbig)

# fill x with coordinates of points
# the x-points are at positions 0, dx, 2dx,  ... 1-dx, 1
x = np.zeros((imax,1))
y = np.zeros((jmax,1))
xplot = np.zeros((imax,jmax))
yplot = np.zeros((imax,jmax))

dx = 1./(imax-2)
ysize = 1.
dy = ysize/(jmax-2)

yshift = 0.5*ysize # shift so that jet centerline is in the middle
for j in range(0,jmax):
   for i in range(0,imax):
      x[i] = (i-0.5)*dx
      y[j] = (j-0.5)*dy - yshift
      xplot[i,j] = x[i]
      yplot[i,j] = y[j]




# initialise arrays Tn1 and To1 are filled with zero's
Tn1  = np.zeros((imax,jmax))
To1  = np.zeros((imax,jmax))
Tn2  = np.zeros((imax,jmax))
To2  = np.zeros((imax,jmax))
K    = np.zeros((imax,jmax))
U    = np.zeros((imax,jmax))
V    = np.zeros((imax,jmax))
S1   = np.zeros((imax,jmax))
S2   = np.zeros((imax,jmax))
Kfilt= np.zeros((imax,jmax))
for j in range(0,jmax):
   for i in range(0,imax):
      if( 0.8 <= x[i] and x[i] <= 0.9 and -0.05 <= y[j] and y[j] <= 0.05):
         S2[i,j] = 1.0

#jet
# half width fillows from tanh(eta)^2 = 0.5
for j in range(0,jmax):
   for i in range(0,imax):
      sigma    = 7.67
      eta_half = 0.88137 # value of eta = sigma*y/x for which (1-tanh(eta)**2) = 0.5
      b_half   = eta_half*(x[i]+VO)/sigma
      eta = sigma*(y[j]-ymp)/(x[i]+VO)
    
      U[i,j] = 0.5*sqrt(3.)*sqrt(Kbig*sigma/(x[i]+VO))*(1-tanh(eta)**2)
      V[i,j] = 0.25*sqrt(3.)*sqrt(Kbig/(sigma*(x[i]+VO)))*(2*eta*(1-tanh(eta)**2)-tanh(eta))

      if(abs(y[j]-ymp)<timeswidtheps*b_half):
         Ucl = 0.5*sqrt(3.)*sqrt(Kbig*sigma/(x[i]+VO))
         epsilon_tau = 0.037*b_half*Ucl
      else:
         epsilon_tau = 0.
      K[i,j] = 2.*epsilon_tau + 0.001

# smooth out K

nfilt = 3
fc  = 0.5
foc = (1.-fc)/4
for ifilt in range(0,nfilt):
   Kfilt = K.copy()
   for j in range(1,jmax-1):
      for i in range(1,imax-1):
          K[i,j] = fc*Kfilt[i,j]+foc*(Kfilt[i-1,j]+Kfilt[i+1,j]+Kfilt[i,j-1]+Kfilt[i,j+1])



   plt.clf()
#  plt.contour(xplot,yplot,Tn1,20)
   ax = plt.axes(projection='3d')
   ax.plot_surface(xplot,yplot,V)
# axis sets limits z-axis
#  ax.set_zlim3d(0,0.5)
#   plt.plot_surface(xplot,yplot,Tn1)
#   surf(xplot,yplot,Tn1)

# pause 0.1 seconds after plotting


plt.show()
