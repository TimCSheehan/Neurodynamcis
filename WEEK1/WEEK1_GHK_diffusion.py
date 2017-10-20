
# coding: utf-8

# In[153]:

import scipy as sp
import scipy.constants
import pylab as plt
from scipy.integrate import odeint
import numpy as np


# In[78]:

## Setup constants
V_t = 26.7      # thermal voltage, in mV
C_m = 1.0       # membrane capacitance, in uF/cm^2

# Free ion concentrations of Na+, K+, and Cl- 
C_Na_o = 145.0  # in mM
C_Na_i = 10.0
C_K_o  = 5.0
C_K_i  = 140.0
C_Cl_o = 110.0
C_Cl_i = 4.0

# Conductances
g_Na = 12.0    # in mS/cm^2
g_K  = 4.0
g_Cl = 0.1

# Nernst reversal potentials
E_Na = V_t * sp.log(C_Na_o / C_Na_i)    # in mV
E_K  = V_t * sp.log(C_K_o  / C_K_i )
E_Cl = V_t * sp.log(C_Cl_i / C_Cl_o)    # reversed since Cl is negatively charged

# Permeabilities
P_Na = g_Na * V_t / E_Na * (1 / C_Na_i - 1 / C_Na_o)    # in mS/(cm^2*uM)
P_K  = g_K  * V_t / E_K  * (1 / C_K_i  - 1 / C_K_o )
P_Cl =-g_Cl * V_t / E_Cl * (1 / C_Cl_i - 1 / C_Cl_o)


# In[4]:

### Part 1: Linear and nonlinear current approximations ###
V_m = sp.arange(-150, 150)   # in mV

# Calculate current densities for all V_m
J_Na_lin = g_Na * (V_m - E_Na)  # in uA/cm^2
J_K_lin  = g_K  * (V_m - E_K )
J_Cl_lin = g_Cl * (V_m - E_Cl)

plt.figure()
plt.plot(V_m, 0*V_m, 'k-')
# recommended style: linear uses dashes, GHK uses solid lines, same color for the same ions
plt.plot(V_m, J_Na_lin, 'r--', label='$J_{Na}$ (linear)')
plt.plot(V_m, J_K_lin,  'g--', label='$J_K$ (linear)')
plt.plot(V_m, J_Cl_lin, 'b--', label='$J_{Cl}$ (linear)')
plt.xlabel('$V_m$ (mV)')
plt.ylabel('$J$ ($\mu A/cm^2$)')
plt.title('Problem 1')
plt.legend(loc='lower right')
plt.show()


# In[127]:

Vm_range = np.arange(-150,150,1)
JK_prime = g_K*(Vm_range-E_K)
JK = P_K*Vm_range*(C_K_i-C_K_o*np.exp(-Vm_range/V_t) )/(1-np.exp(-Vm_range/V_t))

Na_prime = g_Na*(Vm_range-E_Na)
Na = P_Na*Vm_range*(C_Na_i-C_Na_o*np.exp(-Vm_range/V_t) )/(1-np.exp(-Vm_range/V_t))

Cl_prime = g_Cl*(Vm_range-E_K)
Cl = P_Cl*Vm_range*(C_Cl_i-C_Cl_o*np.exp(-Vm_range/V_t) )/(1-np.exp(-Vm_range/V_t))


plt.figure()
plt.plot(V_t*JK/JK,JK , 'k--', label="""$|V_t|$""")
plt.plot(V_t*JK/JK*-1,JK , 'k--')
plt.plot(Vm_range, JK_prime, 'g--', label="""$J_K'$ (linear)""")
plt.plot(Vm_range, JK, 'g', label="""$J_K$""")

plt.xlabel('$V_m$ (mV)')
plt.ylabel('$J$ ($\mu A/cm^2$)')
plt.title('Problem 1 [$k^+$]')
plt.legend(loc='best')

#plt.savefig('Part1_K.png')
plt.show()


# $J'_k$ is a poor approximation for $J_k$ at $V_m$ $\geq$ $V_t$, but when $V_m$ $<<$ $V_t$, $J'_k$ approaches the more accurate estimate $J_k$.
# 

# In[128]:

J_k_high = P_K*Vm_range*(C_K_i)
J_k_low = P_K*Vm_range*(C_K_o)

plt.figure()
plt.plot(V_t*JK/JK,J_k_high , 'k--', label="""$|V_t|$""")
plt.plot(V_t*JK/JK*-1,J_k_high , 'k--')
plt.plot(Vm_range,J_k_high,'g--',label = '$J_k$ [$V_m >> V_t$]')
plt.plot(Vm_range,JK,'g',label = '$J_k$')

plt.xlabel('$V_m$ (mV)')
plt.ylabel('$J$ ($\mu A/cm^2$)')
plt.title('Problem 1 [$k^+$, $V_m$ $>>$ $V_t$]')
plt.legend(loc='best')
#plt.savefig('Part1_K-.png')
plt.show()

plt.figure()
plt.plot(V_t*JK/JK,JK , 'k--', label="""$|V_t|$""")
plt.plot(V_t*JK/JK*-1,JK , 'k--')
plt.plot(Vm_range,J_k_low,'g--',label = '$J_k$ [$V_m << V_t$]')
plt.plot(Vm_range,JK,'g',label = '$J_k$')

plt.xlabel('$V_m$ (mV)')
plt.ylabel('$J$ ($\mu A/cm^2$)')
plt.title('Problem 1 [$k^+$, $V_m$ $<<$ $V_t$]')
plt.legend(loc='best')
#plt.savefig('Part1_K+.png')
plt.show()


# In[129]:

plt.figure()
plt.plot(V_t*Na/Na,Na , 'k--', label="""$|V_t|$""")
plt.plot(V_t*Na/Na*-1,Na , 'k--')
plt.plot(Vm_range, Na_prime,'r--', label="""$J_{Na^+}'$ (linear)""")
plt.plot(Vm_range, Na, 'r', label="""$J_{Na^+}$""")

plt.xlabel('$V_m$ (mV)')
plt.ylabel('$J$ ($\mu A/cm^2$)')
plt.title('Problem 1 [$Na^+$]')
plt.legend(loc='best')
plt.savefig('Part1_Na.png')
plt.show()

plt.figure()
plt.plot(V_t*Na/Na,Cl , 'k--', label="""$|V_t|$""")
plt.plot(V_t*Na/Na*-1,Cl , 'k--')
plt.plot(Vm_range, Cl_prime, 'b--', label="""$J_{Cl^-}'$ (linear)""")
plt.plot(Vm_range, Cl, 'b', label="""$J_{Cl^-}$""")

plt.xlabel('$V_m$ (mV)')
plt.ylabel('$J$ ($\mu A/cm^2$)')
plt.title('Problem 1 [$Cl^-$]')
plt.legend(loc='best')
plt.savefig('Part1_Cl.png')
plt.show()


# In[120]:

### Part 2: The resting potential ###
# Derive these equations. You will either need to write the equations write in
# Microsoft Word or Latex or submit a hard copy of hand-written derivations.

# Computing the resting potentials, in mV
V_r_GHK = V_t*sp.log((P_Na*C_Na_o + P_K*C_K_o + P_Cl*C_Cl_i) / (P_Na*C_Na_i + P_K*C_K_i + P_Cl*C_Cl_o))
V_r_lin = (g_Na*E_Na + g_K*E_K + g_Cl*E_Cl)/(g_Na + g_K + g_Cl)
print(V_r_GHK,V_r_lin)

### Part 3: GHK membrane dynamics ###

t = sp.arange(0.0, 1.0, 0.01)

## Linear Approximation
# Na+ is active for first 500 ms
def g_Na_t(t): return g_Na * (t <= 0.5)

# K+ is active during first 250 ms and last 500 ms
def g_K_t(t):  return g_K  * ((t <= 0.25) + (t >= 0.5))

# Cl- is active during the last 500 ms & first 250 ms
def g_Cl_t(t):  return g_Cl  * ((t <= 0.25) + (t >= 0.5))


# Na+ is active for first 500 ms
def P_Na_t(t): return P_Na * (t <= 0.5)

# K+ is active during first 250 ms and last 500 ms
def P_K_t(t):  return P_K  * ((t <= 0.25) + (t >= 0.5))

# Cl- is active during the last 500 ms & first 250 ms
def P_Cl_t(t):  return P_Cl  * ((t <= 0.25) + (t >= 0.5))


# Time derivative of V_m, as a function of t and V_m
def dVmdt_lin(V_m, t):
    return -1/C_m * (g_Na_t(t)*(V_m-E_Na) + g_K_t(t)*(V_m-E_K) + g_Cl_t(t)*(V_m-E_Cl))

def dVmdt_GHK(V_m, t): # JNa, JK JCl
    return -1/C_m * (P_Na_t(t)*V_m*(C_Na_i-C_Na_o*np.exp(-V_m/V_t) )/(1-np.exp(-V_m/V_t)) 
                     + P_K_t(t)*V_m*(C_K_i-C_K_o*np.exp(-V_m/V_t) )/(1-np.exp(-V_m/V_t)) 
                     + P_Cl_t(t)*V_m*(C_Cl_i-C_Cl_o*np.exp(-V_m/V_t) )/(1-np.exp(-V_m/V_t)))



#P_Na_t = g_Na_t * V_t / E_Na * (1 / C_Na_i - 1 / C_Na_o)    # in mS/(cm^2*uM)
#P_K  = g_K  * V_t / E_K  * (1 / C_K_i  - 1 / C_K_o )
#P_Cl =-g_Cl * V_t / E_Cl * (1 / C_Cl_i - 1 / C_Cl_o)

V_m_lin = odeint(dVmdt_lin, V_r_lin, t)
V_m_GHK = odeint(dVmdt_GHK,V_r_GHK,t)


plt.figure(figsize=(6,12))

plt.subplot(2,1,1) # top 1/2 panel, V_m
plt.plot(t, 0*t, 'k:')
# recommended style: be consistent! linear uses dashes, GHK uses solid lines
plt.plot(t, V_m_lin, 'k--', label='linear')
plt.plot(t, V_m_GHK, 'k', label='GHK')
plt.ylabel('$V_m$ (mV)')
plt.title('Problem 3');
plt.legend(loc='upper right')

plt.subplot(6,1,4) # bottom 4/6 panel, g_K
plt.plot(t, g_K_t(t), color='black')
plt.ylim(-0.1, g_K+0.2)
plt.ylabel('$g_K$ (mS/cm$^2$)')

plt.subplot(6,1,5) # bottom 5/6 panel, g_Na
plt.plot(t, g_Na_t(t), color='black')
plt.ylim(-0.1, g_Na+0.2)
plt.ylabel('$g_{Na}$ (mS/cm$^2$)')

plt.subplot(6,1,6) # bottom 6/6 panel, g_Cl
plt.plot(t, g_Cl_t(t), color='black')
plt.ylim(-0.1, g_Cl+0.2)
plt.ylabel('$g_{Cl}$ (mS/cm$^2$)')

plt.xlabel('Time (s)')

plt.show()
#plt.savefig('Part3.png')


# In[160]:

# part 4
C_m1 = 1.3*10**-6 #F/(cm*cm)
dV = 105*10**-3
dq = C_m1*dV
print(dq) # change in charge
#b 
r = 3*10**-6 # m
r_cm = r*100 # cm
sa_sphere = 4*np.pi*r_cm**2 # surface area of cell

dq_sphere = dq*sa_sphere # change in charge in cell
print(dq_sphere)
elec_in_col = 6.242*10**18
elec_sphere = dq_sphere*elec_in_col
elec_sphere_mol = elec_sphere/sp.constants.Avogadro
print(elec_sphere,elec_sphere_mol)

volume_sphere = 4/3*np.pi*r_cm**3 # volume of cell in cm3
volumeL_sphere = volume_sphere/1000 # volume of cell in L
moles_K = 140*10**-3*volumeL_sphere # molarity * volume = moles
moles_Na = 10*10**-3*volumeL_sphere # molarity * volume = moles

p_chg_K = elec_sphere_mol/moles_K *100
p_chg_Na = elec_sphere_mol/moles_Na *100
print(volumeL_sphere,moles_K,moles_Na,p_chg_K,p_chg_Na)

