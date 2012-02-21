#!/usr/bin/python

import gaussianenv as ge
import poissonneuron as pn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

dt = 0.0001
phi = 1.2
alpha = 0.2
zeta = 1.0
eta = 1.0
gamma = 1.0
timewindow = 10000
dm = 0.2
tau = 0.5

env = ge.GaussianEnv(gamma=gamma,eta=eta,zeta=zeta,x0=0.0,y0=.0,L=1.0,N=1,order=1,sigma=0.1,Lx=1.0,Ly=1.0)
code = pn.PoissonPlasticCode(A=alpha,phi=phi,tau=tau,thetas=np.arange(-2.0,2.0,0.1),dm=dm)

s = np.zeros((timewindow,))
sps = np.zeros((timewindow,code.N))
rates = np.zeros((timewindow,code.N))
for i in range(timewindow):
	s[i] = env.samplestep(dt).ravel()
	rates[i,:] = code.rates(s[i])
	sps[i,:] = code.spikes(s[i],dt)

plt.close()		

plt.plot(np.arange(0.0,dt*timewindow,dt),s,'r')

if sum(sps) !=0:
	spikes = where(sps == 1)
	time = np.arange(0.0,timewindow*dt,dt)
	ts = [time[i] for i in spikes[0]]
	thetas = [code.neurons[i].theta for i in spikes[1]]

ext = (0.0,dt*timewindow,code.neurons[-1].theta,code.neurons[0].theta)
plt.imshow(rates.T,extent=ext,cmap = cm.gist_yarg,aspect = 'auto',interpolation ='nearest')
plt.plot(ts,thetas,'yo')
plt.show()
