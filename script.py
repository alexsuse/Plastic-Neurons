#!/usr/bin/python
"""Particle filtering for nonlinear dynamic systems observed through adaptive poisson neurons
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
particles = np.zeros((timewindow,nparticles))
weights = np.ones((timewindow,nparticles))/nparticles
essthreshold= 2.0/nparticles
dyingrates = np.zeros((nparticles,))

particles[-1,:] = np.random.normal(env.getstate(),eta**2/(2*gamma),particles[-1,:].shape)

for i in range(timewindow):
	s[i] = env.samplestep(dt).ravel()
	rates[i,:] = code.rates(s[i])
	sps[i,:] = code.spikes(s[i],dt)
	particles[i,:] = (1-gamma*dt)*particles[i-1,:]+np.sqrt(dt)*np.random.normal(0.0,eta,nparticles)
	weights[i,:] = weights[i-1,:]
	a = np.where(sps[i,:]==1)[0]
	if a:
		print "spikes"
		liks = code.neurons[a[0]].likelihood(particles[i,:])
		weights[i,:] = weights[i-1,:]*code.neurons[a[0]].likelihood(particles[i,:]) 
		weights[i,:] = weights[i,:]/np.sum(weights[i,:])
	for j in range(nparticles):
		dyingrates[j] = dt*np.sum(code.rates(particles[i,j]))
	c = np.min(dyingrates[:])*dt
	dyingrates =dyingrates-c
	dyingprob= np.sum(dyingrates)
	if np.random.uniform()<dyingprob:
		print "don't die on me!",dyingprob
		dead = pn.choice(dyingrates)
		weights[i,dead]=0.0
		brancher = pn.choice(weights[i,:])
		particles[i,dead]=particles[i,brancher]
		weights[i,dead] = weights[i,brancher]
#		weights[i,:] = weights[i,:]/np.sum(weights[i,:])
		weights[i,:] = 1.0/nparticles
	if np.sum(weights[i,:]*weights[i,:])>essthreshold:
		print "let's shake it!"
		particles[i,:] = particles[i,pn.choice(weights[i,:],shape=particles[i,:].shape)]
		weights[i,:] = 1.0/nparticles	
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
