#!/usr/bin/python
"""Particle filtering for nonlinear dynamic systems observed through adaptive poisson neurons
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def particle_filter(code,env,timewindow=20000,dt=0.001,nparticles=20):
	s = np.zeros((timewindow,))
	sps = np.zeros((timewindow,code.N))
	rates = np.zeros((timewindow,code.N))
	particles = np.zeros((timewindow,nparticles))
	weights = np.ones((timewindow,nparticles))/nparticles
	essthreshold= 2.0/nparticles
	eta = env.geteta()	
	gamma = env.getgamma()
	particles[-1,:] = np.random.normal(env.getstate(),eta**2/(2*gamma),particles[-1,:].shape)
	olda = ""
	
	for i in range(timewindow):
		a = "%2.1f percent "% float(100.0*i/timewindow)
		if a!=olda:
			olda=a
			print a
		s[i] = env.samplestep(dt).ravel()
		[sps[i,:],rates[i,:]] = code.spikes(s[i],dt)
		particles[i,:] = particles[i-1,:]+dt*env.drift(particles[i-1,:])+np.sqrt(dt)*np.random.normal(0.0,eta,nparticles)
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
	times = np.arange(0.0,dt*timewindow,dt)
	
	
	(m,st) = weighted_avg_and_std(particles,weights,nparticles,axis=1)
	mse = np.sum((m-s)**2)/timewindow
	return [m,sps,s,st,mse]


def weighted_avg_and_std(values, ws,nparticles,axis=None):
	average = np.repeat(np.array([np.average(values,weights=ws,axis=1)]).T,nparticles,axis=1)
	variance = np.sum(ws*(values-average)**2,axis=1)  # Fast and numerically precise
	return (average[:,0], np.sqrt(variance))
