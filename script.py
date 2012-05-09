#!/usr/bin/python
"""Particle filtering for nonlinear dynamic systems observed through adaptive poisson neurons
"""
import particlefilter as pf
import gaussianenv as ge
import poissonneuron as pn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#parameter definitions

dt = 0.001
phi = 1.2
alpha = 0.2
zeta = 1.0
eta = 1.8
gamma = 1.2
timewindow = 5000
dm = 0.2
tau = 0.5
nparticles = 20

#env is the "environment", that is, the true process to which we don't have access

env = ge.GaussianEnv(gamma=gamma,eta=eta,zeta=zeta,x0=0.0,y0=.0,L=1.0,N=1,order=1,sigma=0.1,Lx=1.0,Ly=1.0)
env.reset(np.array([0.0]))

#code is the population of neurons, plastic poisson neurons

code = pn.PoissonPlasticCode(A=alpha,phi=phi,tau=tau,thetas=np.arange(-3.0,3.0,0.3),dm=dm)

#s is the stimulus, sps holds the spikes, rates the rates of each neuron and particles give the position of the particles
#weights gives the weights associated with each particle

[mg,varg,spsg,sg] = pf.gaussian_filter(code,env,timewindow=timewindow,dt=dt)

[mp,varp,spsp,sp,mse] = pf.particle_filter(code,env,timewindow=timewindow,dt=dt,nparticles=nparticles)

plt.close()	
plt.figure()
ax1 = plt.gcf().add_subplot(2,1,1)
times = np.arange(0.0,dt*timewindow,dt)

ax1.plot(times,sg,'r')
if sum(sum(spsg)) !=0:
	(ts,neurs) = np.where(spsg == 1)
	spiketimes = times[ts]
	thetas = [code.neurons[i].theta for i in neurs]


ax1.plot(spiketimes,thetas,'yo')
ax1.plot(times,mg,'b')
ax2 = plt.gcf().add_subplot(2,1,2)

ax2.plot(times,sp,'r')
if sum(sum(spsp)) !=0:
	(tsp,neursp) = np.where(spsp == 1)
	spiketimesp = times[tsp]
	thetasp = [code.neurons[i].theta for i in neursp]


ax2.plot(spiketimesp,thetasp,'yo')
ax2.plot(times,mp,'b')
