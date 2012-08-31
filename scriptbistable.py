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
phi = 0.1
alpha = 0.2
eta = 0.85
gamma = 2.0
timewindow = 200000
dm = 0.2
x0 = 1.0
tau = 0.5
nparticles = 10

#env is the "environment", that is, the true process to which we don't have access

env = ge.BistableEnv(gamma=gamma,eta=eta,x0=1.0,order=1,N=1)
env.reset(np.array([0.0]))

#code is the population of neurons, plastic poisson neurons

code = pn.PoissonPlasticCode(A=alpha,phi=phi,tau=tau,thetas=np.arange(-3.0,3.0,0.1),dm=dm,alpha=alpha)

#s is the stimulus, sps holds the spikes, rates the rates of each neuron and particles give the position of the particles
#weights gives the weights associated with each particle

f = lambda x : -1.0+2.0/(1.0+np.exp(-5*x))

[s,mse,frate,m,st,spiketrain,spiketimes] = pf.fast_particle_filter(code,env,dt=dt,timewindow=timewindow,nparticles=nparticles,mode='v',testf = f)

plt.close()	

times = np.arange(0.0,dt*timewindow,dt)
plt.figure()

ax1 = plt.gcf().add_subplot(1,2,1)
ax1.plot(times,map(f,s),'r')

#m = np.average(particles,weights=weights,axis=1)
#st = np.std(particles,weights=weights,axis=1)
#ext = (0.0,dt*timewindow,code.neurons[-1].theta,code.neurons[0].theta)
#plt.imshow(rates.T,extent=ext,cmap = cm.gist_yarg,aspect = 'auto',interpolation ='nearest')
thetas = [code.neurons[i].theta for i in spiketrain]
ax1.plot(times[spiketimes],map(f,thetas),'yo')
ax1.plot(times,m,'b')
ax1.plot(times,m-st,'k',times,m+st,'k')
ax2 = plt.gcf().add_subplot(1,2,2)
ax2.plot(times,s)
plt.savefig('filtering_bistable_sigmoid.png',dpi=200)


