
# coding: utf-8

# In[1]:

import scipy as sp
import pylab as plt
from scipy.integrate import odeint
import numpy as np
import random


# In[2]:

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

plt.rc('font', **font)


# In[245]:

# Constants
C_m = 1.0
g_Na = 120
g_K = 36.0
g_L = 0.3
E_Na = 45.0
E_K = -82.0
E_L = -59.387
E_Cl = -80.0
E_ex = -38.0

# channel gating kinetics

def a_m(V): return 0.1*(V+45.0)/(1.0-np.exp(-(V+45.0)/10.0))
def b_m(V): return 4.0*np.exp(-(V+70.0)/18.0)
def a_h(V): return 0.07*sp.exp(-(V+70.0) / 20.0)
def b_h(V): return 1.0/(1.0 + sp.exp(-(V+40.0) / 10.0))
def a_n(V): return  0.01*(V+60.0)/(1.0 - sp.exp(-(V+60.0) / 10.0))
def b_n(V): return  0.125*sp.exp(-(V+70) / 80.0)

# def a_m(V): return (25-V)/(10*(np.exp((25-V)/10)-1))
# def b_m(V): return 4*np.exp(-V/18)
# def b_m(V): return 4*np.exp(-V/18)
# def a_h(V): return 0.07*np.exp(-V/20)
# def b_h(V): return 1/(np.exp((30-V)/10)+1)
# def a_n(V): return (10-V)/(100*(np.exp((10-V)/10)-1))
# def b_n(V): return 0.125*np.exp(-V/80)


# In[510]:

# alpha/beta constants for ex/inhib synapses
a_r_i = 5.0
b_r_i = 0.18
a_r_e = 2.4
b_r_e = 0.56

# [T] equations for synapses
T_max_i = 1.5
T_max_e = 1.0
K_p = 5.0
V_p = 7.0

def T_i(V_pre): return T_max_i/(1+np.exp(-(V_pre-V_p)/K_p)) 
def T_e(V_pre): return T_max_e/(1+np.exp(-(V_pre-V_p)/K_p)) 


# In[247]:

# Differential gating equations
def dm(V,m): return a_m(V)*(1.0-m)-b_m(V)*m
def dh(V,h): return a_h(V)*(1.0-h)-b_h(V)*h
def dn(V,n): return a_n(V)*(1.0-n)-b_n(V)*n
def dr_i(V,r): return a_r_i*T_i(V)*(1.0-r)-b_r_i*r
def dr_e(V,r): return a_r_e*T_e(V)*(1.0-r)-b_r_e*r


# In[248]:

# Membrane currents (uA/cm^2)
def I_Na(V,m,h): return g_Na*(m**3)*h*(V-E_Na)
def I_K(V,n): return g_K*(n**4)*(V-E_K)
def I_L(V): return g_L*(V-E_L)

def I_syn_i(V,g_GABA,r): return sp.array(sp.matrix(r)*sp.matrix(g_GABA))*(V-E_Cl)
def I_syn_e(V,g_Glu,r): return sp.array(sp.matrix(r)*sp.matrix(g_Glu))*(V-E_ex)


# In[374]:

# membrane voltage differential 
def dV(V,m,h,n,r_i,r_e,I_ext,g_GABA,g_Glu): 
    return (-I_Na(V,m,h) -I_K(V,n)-I_L(V)-I_syn_i(V,g_GABA,r_i)-I_syn_e(V,g_Glu,r_e)+I_ext)/C_m
# X- column -> cells, rows -> parameters

# g_GABA (inhibitory synapses) rows -> presynaptic cell, column -> postsynaptic cell
def d_single(X,t,I_ext,g_GABA,g_Glu):
    V,m,h,n,r_i,r_e = X[0,:],X[1,:],X[2,:],X[3,:],X[4,:],X[5,:]
    return sp.array([
            dV(V,m,h,n,r_i,r_e,I_ext(t),g_GABA,g_Glu)[0],
            dm(V,m), dh(V,h), dn(V,n),
            dr_i(V,r_i), 
            dr_e(V,r_e)
        ])


# In[375]:

# multiple Neurons
# I_exts takes time and returns row vector of current applied to each cell
def d(X,t,I_exts,g_GABA,g_Glu):
    return sp.reshape(d_single(sp.reshape(X,(6,int(len(X)//6))),t,I_exts,g_GABA,g_Glu),len(X))

# simulate whole network
def network(I_exts,g_GABA,g_Glu):
    ncells = sp.size(g_GABA,0)
    return sp.reshape(
    odeint(d,sp.repeat([-70,.05,0.6,0.3,0,0],ncells),t,(I_exts,g_GABA,g_Glu)).T,(6,ncells,len(t))
    )


# # PART 1: Uncoupled Neurons

# In[376]:

# PART 1
I_exts_ = np.array((10,20))
nocon2 = sp.zeros((2,2)) # no connection network
def I_exts(t): return I_exts_
t = sp.linspace(0,500,5000) # 0.1 ms / div
X = network(I_exts,nocon2,nocon2)
V = X[0,:,:]
V1 = V[0,:]
V2 = V[1,:]


# In[377]:

t_show = np.arange(4000,5000) # visualize last 100 ms
plt.plot(t[t_show],V1[t_show],label = "neuron 1")
plt.plot(t[t_show],V2[t_show],label = "neuron ")
plt.xlabel('Time (ms)')
plt.ylabel('Vm (mV)')
plt.legend()
# plt.savefig('Part1_spk.png')
plt.show()


# In[272]:

# ISI Functions
def isi(t, V, spike_thresh=0):
    #isi_mean, isi_dev = isi(t, V, spike_thresh=0)
    t = t[1:]
    time = t[ sp.logical_and(V[:-1] < spike_thresh, V[1:] >= spike_thresh)]
    dt = sp.diff(time)
#     print(time)
    return sp.mean(dt), sp.std(dt),dt

def spk_phase(t, V1, V2, spike_thresh=0):
    #phase_mean, isi_mean = spk_phase(t, V1, V2, spike_thresh=0)
    t = t[1:]
    time1 = t[sp.logical_and(V1[:-1] < spike_thresh, V1[1:] >= spike_thresh)]
    time2 = t[sp.logical_and(V2[:-1] < spike_thresh, V2[1:] >= spike_thresh)]
    l = sp.amin([len(time1), len(time2)])
    isi_mean = sp.mean(sp.diff(time1))
    phase_mean = sp.mean((time1[0:l]-time2[0:l]) / isi_mean * 2 * sp.pi)
    return phase_mean, isi_mean


# In[273]:

spk_thresh = 0
IS_1 = isi(t,V1,spk_thresh)
IS_2 = isi(t,V2,spk_thresh)
print('ISI 10uA:',IS_1[0],'+/-',IS_1[1])
print('ISI 20uA:',IS_2[0],'+/-',IS_2[1])


# In[300]:

def sliding_isi(t,vec,win=500,slide_by=100):
    lv = len(vec)
    win_st = np.arange(0,lv-win,slide_by)
    n_samp = len(win_st)
    out = np.zeros(n_samp)
    for i in range(n_samp):
        window = np.arange(win_st[i],win_st[i]+win)
        #if i==0: print(vec[window])
        out[i],_,_ = isi(t[window],vec[window],spk_thresh)
    return out


# In[357]:

plt.plot(IS_1[2],label='Iext = 10uA ')
plt.plot(IS_2[2],label='Iext = 20uA ')
plt.xlabel('Spike #')
plt.ylabel('ISI')
plt.legend()
# plt.savefig('Part1_isi.png')
plt.show()


# # PART 2

# In[305]:

# Network (I_exts,g_GABA,g_Glu)
# g_GABA is a matrix of inhibitory synapses where each row is the presynaptic cell and the
# column is the postsynaptic cell. Same for g_Glu except that it is excitatory
# I_exts is a function handle that accepts the time and returns a row vector of current
# applied to each cell
# OUTPUTS: V,m,h,n, r_i,r_e
GABA_use = 2
g_GABA = np.array([[0, GABA_use],[GABA_use,0]])
X2 = network(I_exts,g_GABA,nocon2)


# In[308]:

V2_1 = X2[0,0,:]
V2_2 = X2[0,1,:]
plt.plot(t,V2_1)
plt.plot(t,V2_2)
plt.show()


# In[512]:

t = sp.linspace(0,500,5000) # 0.1 ms / div
b_r_i = 0.18
I_exts_ = np.array((10,20))
g_GABA_try = np.array([0,0.1,0.2,0.3,0.4,0.5,1.0,1.5,2.0,2.5,3.0,3.5])
#g_GABA_try = np.array([0,0.1,0.5])
n_try = len(g_GABA_try)
ISIs = np.zeros((2,n_try))
all_misis = np.zeros((2,n_try))
all_sisis = np.zeros((2,n_try))
XX2 = []
for i in range(n_try):
    g_GABA = np.array([[0, g_GABA_try[i]],[g_GABA_try[i],0]])
    X0 = network(I_exts,g_GABA,nocon2)
    XX2.append(X0)
    all_misis[:,i], all_sisis[:,i] = get_ISIs(t,X0)


# In[336]:

def get_ISIs(t,X0,win=np.arange(2500,5000)):
    n_elec = np.size(X0,1)
    this_t = t[win]
    misis = np.zeros(n_elec)
    sisis = np.zeros(n_elec)
    for i in range(n_elec):
        this_vec = X0[0,i,win]
        m_isi,s_isi,_ = isi(this_t,this_vec)
        misis[i] = m_isi
        sisis[i] = s_isi
    return misis,sisis
    


# In[518]:

plt.errorbar(g_GABA_try,all_misis[0,:],all_sisis[0,:],label='Iext = 10uA ')
plt.errorbar(g_GABA_try,all_misis[1,:],all_sisis[1,:],label='Iext = 20uA ')
plt.xlabel('GABA Conductance')
plt.ylabel('InterSpike Interval (ms)')
plt.legend()
#plt.savefig('Part2_isi.png')
plt.show()
plt.plot(g_GABA_try,1000/all_misis[0,:],label='Iext = 10uA ')
plt.plot(g_GABA_try,1000/all_misis[1,:],label='Iext = 20uA ')

plt.xlabel('GABA Conductance')
plt.ylabel('Spike Rate (Hz)')
plt.legend()
#plt.savefig('Part2_SR.png')
plt.show()


# In[589]:

# need to add example plots for a few different GABA values of V(t) and r(t) +  freq vs. GABA
# all_misis[0,:]-all_misis[1,:]
# g_GABA_try[4:7]
# g_plot = [0, 5, 10]
v_use = 7
plt.figure(figsize = (10,4) )
plt.subplot(1,2,1)
plt.plot(t[win],XX2[v_use][0,0,win],label = 'Iext  = 10uA')
plt.plot(t[win],XX2[v_use][0,1,win],label = 'Iext  = 20uA')
plt.title('g_GABA ' + str(g_GABA_try[v_use]))
#plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
#plt.savefig('part2_Vt_' + str(g_GABA_try[v_use]) +'.png' )
#plt.show()

plt.subplot(1,2,2)
plt.plot(t[win],XX2[v_use][4,0,win],label = 'Iext  = 10uA')
plt.plot(t[win],XX2[v_use][4,1,win],label = 'Iext  = 20uA')
plt.title('g_GABA ' + str(g_GABA_try[v_use]))
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('r_i')
nam  = 'part2_rt_' + str(int(g_GABA_try[v_use]*10)) +'.png'
print(nam)
#plt.savefig(nam)
plt.show()



# # Part 3 In-Phase Oscillations

# In[483]:

def dr_i_3(V,r,b_r_i): return a_r_i*T_i(V)*(1-r)-b_r_i*r
def d_single_3(t,x,b_r): 
    mat = sp.array([
        dV(x[0,:],x[1,:],x[2,:],x[3,:],x[4,:],[0, 0],I_exts_,inv_i2,nocon2)[0],
         dm(x[0,:],x[1,:]),
         dh(x[0,:],x[2,:]),
         dn(x[0,:],x[3,:]),
         dr_i_3(x[0,:],x[4,:],b_r)
         ])
    return mat

def d_3(X,t,b_r): 
    foo = d_single_3(t,np.reshape(X,(5,2)),b_r)
    return np.reshape(foo,(10,))

def network_3(b_r):
    ncells = 2
    foo = odeint(d_3,sp.repeat([-70,.05,0.6,0.3,0],ncells),t,(b_r)).T
    return sp.reshape(foo,(5,ncells,len(t)))


# In[484]:

GABA_use = 1
I_exts_ = np.array((10,10.1))
b_r_i = 0.18 # want to vary from 0.5 to 0.1
nocon2 = sp.zeros((2,2))
inv_i2 = sp.array([[0,1],[1,0]])
g_GABA = np.array([[0, GABA_use],[GABA_use,0]])
b_r = 0.4, # , makes it a tuple
type(b_r)
bar = network_3(b_r)


# In[502]:

brs = np.arange(0.5,0,-0.1)
n_b = len(brs)
p = np.zeros(n_b)
XX = []
# spk_phase(t, V1, V2, spike_thresh=0
for i in range(n_b):
    X3 = network_3((brs[i],))
    XX.append(X3)
    spk_win,_ = spk_phase(t[win],X3[0,0,win],X3[0,1,win],0)
    p[i] = spk_win
    #print(i)


# In[507]:

# help(odeint)
plt.plot(brs,p)
plt.xlabel('Backward Rate Constant')
plt.ylabel('Phase Offset (rad)')
plt.title('In-Phase Oscillations')
# plt.savefig('Part3_phaseOffset')
plt.show()


# In[571]:

v_use = 4
plt.figure(figsize = (10,4) )
plt.subplot(1,2,1)
plt.plot(t[win],XX[v_use][0,0,win],label = 'Iext  = 10uA')
plt.plot(t[win],XX[v_use][0,1,win],label = 'Iext  = 10.1uA')
plt.title('b_r ' + str(brs[v_use]))
#plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
#plt.savefig('part3_Vt_' + str(brs[v_use]) +'.png' )
#plt.show()

plt.subplot(1,2,2)
plt.plot(t[win],XX[v_use][4,0,win],label = 'Iext  = 10uA')
plt.plot(t[win],XX[v_use][4,1,win],label = 'Iext  = 10.1uA')
plt.title('b_r ' + str(brs[v_use]))
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('r_i')
plt.savefig('part3_VR_' + str(int(brs[v_use]*10)) +'.png')
plt.show()


# # Part 4 Excitatory Synapse Model

# In[635]:

# OUTPUTS: V,m,h,n, r_i,r_e
# g_GABA (inhibitory synapses) rows -> presynaptic cell, column -> postsynaptic cell
gGlu_use = np.arange(0,0.55,0.05) # 0 -> 0.5
I_exts_ = np.array((10,0))
n_glu = len(gGlu_use)
X4 = []
rate1 = np.zeros(n_glu)
rate2 = np.zeros(n_glu)

for i in range(n_glu):
    gGlu = np.array([[0, gGlu_use[i]],[0,0]])
    XX = network(I_exts,nocon2,gGlu)
    X4.append(XX)
    rate1[i],_,_ = isi(t[win],X4[i][0,0,win])
    rate2[i],_,_ = isi(t[win],X4[i][0,1,win])


# In[644]:

plt.plot(gGlu_use,1000/rate1,label = 'Iext = 10uA')
plt.plot(gGlu_use,1000/rate2,label = 'Iext = None')
plt.xlabel('Excitatory (Glutaminergic) Conductance')
plt.ylabel('Firing Rate (Hz)')
plt.legend()
# plt.savefig('part4_fr_gGlu.png')
plt.show()


# In[654]:

v_use = 10
plt.figure(figsize = (10,4) )
plt.subplot(1,2,1)
plt.plot(t[win],X4[v_use][0,0,win],label = 'Iext  = 10uA')
plt.plot(t[win],X4[v_use][0,1,win],label = 'Iext  = 0')
plt.title('g_Glu ' + str(gGlu_use[v_use]))
#plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
#plt.savefig('part3_Vt_' + str(brs[v_use]) +'.png' )
#plt.show()

plt.subplot(1,2,2)
plt.plot(t[win],X4[v_use][4,0,win],label = 'Iext  = 10uA')
plt.plot(t[win],X4[v_use][4,1,win],label = 'Iext  = 0')
plt.title('g_Glu ' + str(gGlu_use[v_use]))
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('r_i')
nam = 'part4_VR_' + str(int(gGlu_use[v_use]*10)) +'.png'
print(nam)
plt.savefig(nam)
plt.show()


# # Part 5: Feedforward Inhibition

# In[670]:

# OUTPUTS: V,m,h,n, r_i,r_e
# g_GABA (inhibitory synapses) rows -> presynaptic cell, column -> postsynaptic cell
# VARYING G_Glu
gGlu_use = np.arange(0,1,0.05) # 0 -> 0.5
gGABA_use = 0.5
I_exts_ = np.array((10,0,0)) # 3 neurons
n_glu = len(gGlu_use)

rate5_1 = np.zeros(n_glu)
rate5_2 = np.zeros(n_glu)
rate5_3 = np.zeros(n_glu)
X5 = []
for i in range(n_glu):
    gGlu = np.array([[0, gGlu_use[i], gGlu_use[i]],[0,0,0],[0,0,0]])
    gGABA = np.array([[0,0,0],[0,0,gGABA_use],[0,0,0]])
    XX = network(I_exts,gGABA,gGlu)
    X5.append(XX)
    rate5_1[i],_,_ = isi(t[win],X5[i][0,0,win])
    rate5_2[i],_,_ = isi(t[win],X5[i][0,1,win])
    rate5_3[i],_,_ = isi(t[win],X5[i][0,2,win])


# In[708]:

gGlu_use = np.arange(0,1,0.05) # 0 -> 0.5
plt.plot(gGlu_use,1000/rate5_1,label = 'I_ext = 10 uA')
plt.plot(gGlu_use,1000/rate5_2,label = 'I_ext = 0')
plt.plot(gGlu_use,1000/rate5_3,label = 'I_ext = 0')
plt.xlabel('G_Glu')
plt.ylabel('Spike Rate (Hz)')
plt.legend()
plt.title('FFI Varying G_Glu (gGABA = 0.5) ')
plt.savefig('part5_SR_gGlu.png')
plt.show()


# In[679]:

#gGlu_use = np.arange(0,1,0.05) # 0 -> 0.5
gGlu_def = 0.4
gGABA_use = np.arange(0,1,0.05)
I_exts_ = np.array((10,0,0)) # 3 neurons
n_gaba = len(gGABA_use)
X6 = []
rate6_1 = np.zeros(n_glu)
rate6_2 = np.zeros(n_glu)
rate6_3 = np.zeros(n_glu)

for i in range(n_glu):
    gGlu = np.array([[0, gGlu_def, gGlu_def],[0,0,0],[0,0,0]])
    gGABA = np.array([[0,0,0],[0,0,gGABA_use[i]],[0,0,0]])
    XX = network(I_exts,gGABA,gGlu)
    X6.append(XX)
    rate6_1[i],_,_ = isi(t[win],X6[i][0,0,win])
    rate6_2[i],_,_ = isi(t[win],X6[i][0,1,win])
    rate6_3[i],_,_ = isi(t[win],X6[i][0,2,win])


# In[ ]:




# In[706]:

gGABA_use = np.arange(0,1,0.05)
plt.plot(gGABA_use,1000/rate6_1,label = 'I_ext = 10 uA')
plt.plot(gGABA_use,1000/rate6_2,label = 'I_ext = 0')
plt.plot(gGABA_use,1000/rate6_3,label = 'I_ext = 0')
plt.xlabel('G_GABA')
plt.ylabel('Spike Rate (Hz)')
plt.legend()
plt.title('FFI Varying g_GABA (g_Glu = 0.4)')
plt.savefig('part5_SR_gGABA.png')
plt.show()


# In[705]:

# vary currents 
gGABA_use = 0.4
gGlu_use = 0.4
I_ext_use = np.arange(5,30,5)
n_Iext = len(I_ext_use)
X7 = []
rate7_1 = np.zeros(n_Iext)
rate7_2 = np.zeros(n_Iext)
rate7_3 = np.zeros(n_Iext)

for i in range(n_Iext):
    gGlu = np.array([[0, gGlu_use, gGlu_use],[0,0,0],[0,0,0]])
    gGABA = np.array([[0,0,0],[0,0,gGABA_use],[0,0,0]])
    I_exts_ = (I_ext_use[i],0,0)
    XX = network(I_exts,gGABA,gGlu)
    X7.append(XX)
    rate7_1[i],_,_ = isi(t[win],X7[i][0,0,win])
    rate7_2[i],_,_ = isi(t[win],X7[i][0,1,win])
    rate7_3[i],_,_ = isi(t[win],X7[i][0,2,win])


# In[738]:

v_use = 1
# plt.figure(figsize = (10,4) )
# plt.subplot(1,2,1)
I_ext_use = np.arange(5,30,5)
plt.plot(t[win],X7[v_use][0,0,win],label = 'Iext  = ' +str(int(I_ext_use[v_use])) +'uA')
plt.plot(t[win],X7[v_use][0,1,win],label = 'Neuron 2')
plt.plot(t[win],X7[v_use][0,2,win],label = 'Neuron 2')
plt.title('I_ext ' + str(I_ext_use[v_use]))
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')

nam = 'part5_V_' + str(int(I_ext_use[v_use])) +'.png'
print(nam)
plt.savefig(nam)
plt.show()


# # Part 6: Feedback Inhibition

# In[709]:

gGABA_use = 0.4
gGlu_use = 0.4
I_ext_use = np.arange(2,30,4)
n_Iext = len(I_ext_use)
X8 = []
rate8_1 = np.zeros(n_Iext)
rate8_2 = np.zeros(n_Iext)
rate8_3 = np.zeros(n_Iext)

for i in range(n_Iext):
    gGlu = np.array([[0, 0, gGlu_use],[0,0,0],[0,gGlu_use,0]])
    gGABA = np.array([[0,0,0],[0,0,gGABA_use],[0,0,0]])
    I_exts_ = (I_ext_use[i],0,0)
    XX = network(I_exts,gGABA,gGlu)
    X8.append(XX)
    rate8_1[i],_,_ = isi(t[win],X8[i][0,0,win])
    rate8_2[i],_,_ = isi(t[win],X8[i][0,1,win])
    rate8_3[i],_,_ = isi(t[win],X8[i][0,2,win])


# In[743]:

I_ext_use = np.arange(2,30,4)
plt.plot(I_ext_use,1000/rate8_1,label = 'Neuron 1')
plt.plot(I_ext_use,1000/rate8_2,label = 'Neuron 2')
plt.plot(I_ext_use,1000/rate8_3,label = 'Neuron 2')
plt.ylabel('Spike Rate (Hz)')
plt.xlabel('I_ext (uA)')
plt.legend()
plt.savefig('part6_Iext.png')
plt.show()


# In[734]:

v_use = 1
I_ext_use = np.arange(2,30,4)
plt.plot(t[win],X8[v_use][0,0,win],label = 'Iext  = ' + str(I_ext_use[v_use]) + 'uA')
plt.plot(t[win],X8[v_use][0,1,win],label = 'Neuron 2')
plt.plot(t[win],X8[v_use][0,2,win],label = 'Neuron 2')
plt.title('I_ext ' + str(I_ext_use[v_use]))
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
nam = 'part6_V_' + str(int(I_ext_use[v_use])) +'.png'
print(nam)
plt.savefig(nam)
plt.show()


# In[ ]:



