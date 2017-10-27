import time
import scipy as sp
import pylab as plt
from scipy.integrate import odeint

#
#           Nonlinear vs. linear pendulum problem
#           -------------------------------------
#
# The equation of motion for a pendulum is
#
# d^2(theta)/dt^2 + sin(theta) = 0
#
# where gravity points "downwards" and theta is the angle the pendulum
# forms with its rest position. This is a nonlinear equation which we will
# solve in the second part. 
#
# Typically, when you learn about pendulums in intro physics, you learn
# the linear approximation, where
#
# d^2(theta)/dt^2 + theta = 0
#
# This is called the small angle approximation, where for small angles,
# sin(theta)=theta. This is the first term of the Taylor expansion of
# sin(theta) around an angle of zero. This is valid for small swings around
# the axis, much like the linear approximation of GHK is close for some
# portion of the curve.
#
# Let's solve and graph the evolution of the linear pendulum over time.
# Since this equation is linear and easily solvable, we know the solution
# to this is a simple harmonic oscillator (think back to solving a spring's
# motion, F=kx)
#
# The solution to such a system is in the form theta = A*cos(omega*t+phi),
# where A is the amplitude of the oscillation, omega is the angular
# frequency (omega=2*pi*f), and phi is the initial displacement.


t_sim = 50              # solve for 50 seconds
t = sp.arange(0, t_sim, 0.2)
theta_0 = 5*sp.pi/180   # initial angle is 5 degrees in radians


# The max amplitude (A) will be the starting amplitude in this system and 
# phi = 0 since we are assuming no initial velocity. Omega = 1 from the 
# condition that d^2(theta)/dt^2 = -theta

A = theta_0
omega = 1
phi = 0

start = time.clock()    # start a stopwatch to see how long this takes

theta_lin = A*sp.cos(omega*t+phi)

time_linear = time.clock() - start

plt.figure()
plt.plot(t, theta_lin)
plt.show()


## There is no elementary form to the solution so we'll have to compute it
# ourselves. We can reduce the second order equation to two coupled first 
# order equations by setting omega = d(theta)/dt, and solve from there.

t_sim = 50              # solve for 50 seconds
omega_0 = 0             # initial velocity = 0

def dthetadt(X, t):
    theta, omega = X
    
    dtheta_dt = omega
    domega_dt = -sp.sin(theta)

    return dtheta_dt, domega_dt
    
start = time.clock()

theta_nonlin = odeint(dthetadt, [theta_0, omega_0], t)

time_nonlin = time.clock() - start

plt.figure()
plt.plot(t, theta_nonlin[:,0])
plt.show()


## What is the time savings you get using the linear approximation?

time_ratio = time_nonlin / time_linear

# Nonlinear equations get very computationally intensive very quickly. If 
# you increase or decrease t_sim, do you get differences in this ratio?


## Comparison of accuracy of methods
#
# In the case of 5 degrees, the linear approximation was pretty good. 
# What about 50 degrees? Could you use it then? What about extreme angles 
# like 150 degrees? You can put both graphs on the same plot with the 
# following lines to compare them.

plt.figure()
plt.plot(t, theta_lin)
plt.plot(t, theta_nonlin[:,0])
plt.show()
