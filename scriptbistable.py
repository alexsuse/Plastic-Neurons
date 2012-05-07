#!/usr/bin/python
"""Particle filtering for nonlinear dynamic systems observed through adaptive poisson neurons
"""
import gaussianenv as ge
import poissonneuron as pn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#parameter definitions

dt = 0.001
phi = 1.2
alpha = 0.1
zeta = 4.0
eta = 1.4
gamma = 1.0
timewindow = 20000
dm = 0.2
tau = 0.5
nparticles = 10

#env is the "environment", that is, the true process to which we don't have access

env = ge.BistableEnv(gamma=gamma,eta=eta,zeta=zeta,x0=0.0,y0=.0,L=1.0,N=1,order=1,sigma=0.1,Lx=1.0,Ly=1.0)
env.reset(np.array([0.0]))

#code is the population of neurons, plastic poisson neurons

code = pn.PoissonPlasticCode(A=alpha,phi=phi,tau=tau,thetas=np.arange(-3.0,3.0,0.1),dm=dm)

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
	print "%f percent" % float(100.0*i/timewindow)
	s[i] = env.samplestep(dt).ravel()
	[sps[i,:],rates[i,:]] = code.spikes(s[i],dt)
	particles[i,:] = (1-gamma*dt)*particles[i-1,:]+np.sqrt(dt)*np.random.normal(0.0,eta,nparticles)
	weights[i,:] = weights[i-1,:]
	a = np.where(sps[i,:]==1)[0]
	if a:
#		print "spikes"
		liks = code.neurons[a[0]].likelihood(particles[i,:])
		weights[i,:] = weights[i-1,:]*liks
		weights[i,:] = weights[i,:]/np.sum(weights[i,:])
	else:
#		print "no spikes"
		rt = np.array([code.totalrate(j) for j in particles[i,:]])
		weights[i,:] = weights[i-1,:]*(1.0-rt*dt)
		weights[i,:] = weights[i,:]/np.sum(weights[i,:])
	if np.sum(weights[i,:]*weights[i,:])>essthreshold:
#		print "let's shake it!"
		particles[i,:] = particles[i,pn.choice(weights[i,:],shape=particles[i,:].shape)]
		weights[i,:] = 1.0/nparticles	
plt.close()	

times = np.arange(0.0,dt*timewindow,dt)

plt.plot(times,s,'r')
if sum(sum(sps)) !=0:
	(ts,neurs) = np.where(sps == 1)
	spiketimes = times[ts]
	thetas = [code.neurons[i].theta for i in neurs]

m = np.average(particles,weights=weights,axis=1)

#ext = (0.0,dt*timewindow,code.neurons[-1].theta,code.neurons[0].theta)
#plt.imshow(rates.T,extent=ext,cmap = cm.gist_yarg,aspect = 'auto',interpolation ='nearest')
plt.plot(spiketimes,thetas,'yo')
plt.plot(times,m,'b')
