import pylab as plt
import scipy as sp
import scipy.linalg as lin
from ndtools.clines import *

# Pendulum Problems
#
# Next we will do an analysis of the damped nonlinear pendulum. This is a
# two-dimensional system that is closely related to the nonlinear pendulum 
# presented above. The equations that describe this system are:
#
#    Dx = -b*x -sin(y-a)
#    Dy = x
#
# Start with initial conditions: 
#    
#    x(0) = 0
#    y(0) = 0
#
# Then play around with them to see the range of dynamics.
#
# ** For the problems, read through and run the code below see examples of the 
# getNullclines, getCrossings and getJacobian which are included in the
# ndtools.clines module.
#
#
# 1) Use meshgrid and quiver to plot the gradient of the nonlinear 
# pendulum system as a function of x and y, using bounds:  
#
#    -2  <=  x  <=  2
#    -pi <=  y  <=  pi
#
# Look at the gradient for the following parameters:
#
#    b = 0;  a = 0;
#    b = 0;  a = pi/2;
#    b = 0;  a = pi;
#    b = 1;  a = pi/2;
#    b = 1;  a = pi;
#
# 2) Program the nonlinear pendulum. Solve it for each of the above
# conditions, and plot the result for: 
#
#     0  <=  t  <= 100.
#
# 3) Use getNullclines to get a pair of null-clines for two undamped conditions:
#
#    b = 0,  a = 0
#    b = 0,  a = pi
#
# Try these and other conditions. The code identifies only a single pair of
# null-clines for each dynamic variable. If there is a second, take a
# moment and identify it. 
#
# For the Homework, you will need to identify fixed-points. To do this
# using the contours, try using getCrossings.
#
# 4) Try linearizing the system around <x,y> = <0,0> for:
#
#    b = 0;  a = 0
#
# using getJacobian. Compute the eigenvalues of this matrix, using the eig 
# function or the SVD function. Make sure the eigenvalues confirm the 
# dynamics of the gradient map. 


# Fixed-points and null-clines for the simple harmonic oscillator.
#
#    Dx  = -w*y
#    Dy  =  w*x
#
# Where w is the oscillator frequency.

# 2D-axis specification
N = 25
x = sp.linspace(-1,1,N)
y = sp.linspace(-1,1,N)
X, Y = sp.meshgrid(x,y)

# Constants
w = 1 # the oscillator frequency

# The simple harmonic oscillator
DX = -w*Y
DY =  w*X

# Null-clines and fixed-points
# You can also derive the null-cline equations for pendulums themselves...
# Then use fmin() to get the fixed points
nc1_x, nc1_y = getNullcline(DX,x,y)
nc2_x, nc2_y = getNullcline(DY,x,y)
fp_x, fp_y = getCrossings(nc1_x,nc1_y, nc2_x,nc2_y)

# Make sure we have only the first fixed point
fp_x, fp_y = fp_x[0], fp_y[0]

# Linearization
J = getJacobian(X,Y,DX,DY,fp_x,fp_y)
[eig_val, eig_vec] = lin.eig(J)
print 'Eigenvalues =', eig_val


# Plotting the null-clines and fixed-points
plt.figure();
plt.quiver(X,Y,DX,DY, scale=20, label='Gradient') 
plt.plot(nc1_x, nc1_y, 'k', label='F(x,y) = 0')
plt.plot(nc2_x, nc2_y, 'k', label='G(x,y) = 0')
plt.plot(fp_x,  fp_y,  'ro', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')
plt.title('Null Clines and Fixed Points of the simple harmonic oscillator');
plt.show()
