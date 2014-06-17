#!/usr/bin/python
"""
Gaussian filtering for the OU process observed through adaptive poisson neurons with gaussian tuning functions.
"""
import gaussianenv as ge
import poissonneuron as pn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#parameter definitions

dt = 0.0001
phi = 1.2
alpha = 0.1
zeta = 1.0
eta = 1.0
gamma = 5.0
timewindow = 20000
dm = 0.2
tau = 0.5
nparticles = 5

#env is the "environment", that is, the true process to which we don't have access

env = ge.GaussianEnv(gamma=gamma,eta=eta,zeta=zeta,x0=0.0,y0=.0,L=1.0,N=1,order=1,sigma=0.1,Lx=1.0,Ly=1.0)
env.reset(np.array([0.0]))

#code is the population of neurons, plastic poisson neurons

code = pn.PoissonPlasticCode(A=alpha,phi=phi,tau=tau,thetas=np.arange(-2.0,2.0,0.1),dm=dm)

#s is the stimulus, sps holds the spikes, rates the rates of each neuron and particles give the position of the particles
#weights gives the weights associated with each particle

s = np.zeros((timewindow,))
sps = np.zeros((timewindow,code.N))
rates = np.zeros((timewindow,code.N))
mu = np.zeros((timewindow,1))
sigma = np.zeros((timewindow,1))
mugauss = np.zeros((timewindow,1))
siggauss= np.zeros((timewindow,1))

for i in range(timewindow):
	s[i] = env.samplestep(dt).ravel()
	[sps[i,:],rates[i,:]] = code.spikes(s[i],dt)
	if sum(sps[i,:]) == 0:
		dmuplast = phi*np.dot(km,sigma[i-1,:]*(mu[i-1,:]-code.neurons[:].theta))/sqrt(1.0+sigma[i-1,:]/alpha^2)
		disgplast = 
		mugauss[i,:] = mugauss[i-1,:] - dt*gamma*mugauss[i-1,:]

plt.close()		

plt.plot(np.arange(0.0,dt*timewindow,dt),s,'r')

if sum(sps) !=0:
	spikes = where(sps == 1)
	time = np.arange(0.0,timewindow*dt,dt)
	ts = [time[i] for i in spikes[0]]
	thetas = [code.neurons[i].theta for i in spikes[1]]

m = np.average(particles,weights=weights,axis=1)

ext = (0.0,dt*timewindow,code.neurons[-1].theta,code.neurons[0].theta)
plt.imshow(rates.T,extent=ext,cmap = cm.gist_yarg,aspect = 'auto',interpolation ='nearest')
plt.plot(ts,thetas,'yo')
plt.plot(np.arange(.0,timewindow*dt,dt),m,'b')
plt.show()
