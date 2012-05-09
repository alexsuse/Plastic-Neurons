#!/usr/bin/python
"""Particle filtering for nonlinear dynamic systems observed through adaptive poisson neurons
"""
import gaussianenv as ge
import poissonneuron as pn
import numpy as np
import matplotlib.pyplot as plt
import particlefilter as pf
from matplotlib import cm

#parameter definitions

dt = 0.001
phi = 1.2
alpha = 0.5
zeta = 4.0
eta = 1.4
gamma = 1.0
timewindow = 20000
dm = 0.2
tau = 0.5
nparticles = 20

#env is the "environment", that is, the true process to which we don't have access

env = ge.BistableEnv(gamma=gamma,eta=eta,zeta=zeta,x0=0.0,y0=.0,L=1.0,N=1,order=1,sigma=0.1,Lx=1.0,Ly=1.0)
env.reset(np.array([0.0]))

#code is the population of neurons, plastic poisson neurons

code = pn.PoissonPlasticCode(A=alpha,phi=phi,tau=tau,thetas=np.arange(-3.0,3.0,0.4),dm=dm)

#s is the stimulus, sps holds the spikes, rates the rates of each neuron and particles give the position of the particles
#weights gives the weights associated with each particle

[m,st,sps,s,mse] = pf.particle_filter(code,env,dt=dt,timewindow=timewindow,nparticles=nparticles)

plt.close()	

times = np.arange(0.0,dt*timewindow,dt)

plt.plot(times,s,'r')
if sum(sum(sps)) !=0:
	(ts,neurs) = np.where(sps == 1)
	spiketimes = times[ts]
	thetas = [code.neurons[i].theta for i in neurs]

#m = np.average(particles,weights=weights,axis=1)
#st = np.std(particles,weights=weights,axis=1)
#ext = (0.0,dt*timewindow,code.neurons[-1].theta,code.neurons[0].theta)
#plt.imshow(rates.T,extent=ext,cmap = cm.gist_yarg,aspect = 'auto',interpolation ='nearest')
plt.plot(spiketimes,thetas,'yo')
plt.plot(times,m,'b')
plt.plot(times,m-st,'k',times,m+st,'k')
