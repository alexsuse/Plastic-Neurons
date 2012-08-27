#!/usr/bin/python
"""Particle filtering for nonlinear dynamic systems observed through adaptive poisson neurons
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def choice(p,a=None,shape=(1,),randomstate=np.random):
	"""chooses an element from a with probabilities p. Can return arbitrarily shaped-samples through the shape argument.
	p needs not be normalized, as this is checked for."""
	x = randomstate.uniform(size=shape)
	cump = np.cumsum(p)
	if cump[-1]!=1:
		cump=cump/cump[-1]
	idxs = np.searchsorted(cump,x)
	if a==None:
		return idxs
	else:
		return a[idxs]

def gaussian_filter(code,env,timewindow=20000,dt=0.001,mode='Silent'):
	s = np.zeros((timewindow,))
	s = env.samplestep(dt,N=timewindow).ravel()
	sps = np.zeros((timewindow,code.N))
	rates = np.zeros((timewindow,code.N))
	m = np.zeros((timewindow,))
	sigma = np.ones((timewindow,))
	gamma = env.getgamma()
	eta = env.geteta()
	phi = code.neurons[0].phi
	alpha = code.A	
	olda = ""
	for i in range(timewindow):
		if mode!='Silent':
			percent = "%2.1f percent "% float(100.0*i/timewindow)
			if percent!=olda:
				olda=percent
				print "gaussian filter:"+percent
		[temp1,temp2] = code.spikes(s[i],dt)
		sps[i,:] = np.array(temp1)
		rates[i,:] = np.array(temp2).ravel()
		a = np.where(sps[i,:]==1)[0]
		muterm= 0.0
		sigterm = 0.0
		if a:
			m[i] = (alpha**2*m[i-1]+sigma[i-1]*code.neurons[a].theta)/(alpha**2+sigma[i-1])	
			sigma[i] = sigma[i-1]*alpha**2/(alpha**2+sigma[i-1])
		else:
			for n in code.neurons:
				km = n.getmu()
				cm = np.exp(-0.5*(m[i-1]-n.theta)**2/(alpha**2+sigma[i-1]))
				muterm = muterm + km*cm*(m[i-1]+n.theta)
				sigterm = sigterm + km*cm*(1.0-(m[i-1]+n.theta)**2/(alpha**2+sigma[i-1]))
			vorfaktor = sigma[i-1]*phi/(np.sqrt(1.0+sigma[i-1]/alpha**2)*(alpha**2+sigma[i-1]))
			m[i] = m[i-1] + dt*env.drift(m[i-1])+dt*muterm*vorfaktor
			sigma[i] = sigma[i-1]+dt*env.vardrift(sigma[i-1])+dt*sigma[i-1]*sigterm*vorfaktor
	mse = np.average((m-s)**2)
	return [m,sigma,sps,s,mse]	


def particle_filter(code,env,timewindow=20000,dt=0.001,nparticles=20,mode='Silent',randomstate=np.random,testf = (lambda x: x)):

	s = np.zeros((timewindow,))
	s = env.samplestep(dt,N=timewindow).ravel()
	sps = np.zeros((timewindow,code.N))
	rates = np.zeros((timewindow,code.N))
	thets = [n.theta for n in code.neurons]
	exponent = np.tile(s,(code.N,1))-np.tile(thets,(timewindow,1)).T
	particles = np.zeros((timewindow,nparticles))
	weights = np.ones((timewindow,nparticles))/nparticles
	
	grates = np.exp(-0.5*exponent**2/code.alpha**2).T
	essthreshold= 2.0/nparticles
	eta = env.geteta()	
	gamma = env.getgamma()
	
	particles[-1,:] = randomstate.normal(env.getstate(),eta**2/(2*gamma),particles[-1,:].shape)
	olda = ""
	i=-1
	for stim in s:
		i+=1
		if mode!='Silent':
			percent = "%2.1f percent "% float(100.0*i/timewindow)
			if percent!=olda:
				olda=percent
				print "particle filter:"+percent
			#	print "particles"
			#	print particles[i-1,:10]
			#	print "weights"
			#	print weights[i-1,:10]
		#[sps[i,:],rates[i,:]] = code.spikes(s[i],dt)
		[temp1,temp2] = code.spikes(stim,dt,grates=grates[i,:])
		sps[i,:] = temp1
		rates[i,:] = temp2.ravel()
		particles[i,:] = particles[i-1,:]+dt*env.drift(particles[i-1,:])+np.sqrt(dt)*randomstate.normal(0.0,eta,nparticles)
		a = np.where(sps[i,:]==1)[0]
		if a:
			liks = code.neurons[a[0]].likelihood(particles[i,:])
			weights[i,:] = weights[i-1,:]*liks
			if np.sum(weights[i,:])==0.0:
				print "DANGER, DANGER"
				print np.sum(weights[i,:])
				print liks
				print weights
				weights[i,:] = 1.0/nparticles
			weights[i,:] = weights[i,:]/np.sum(weights[i,:])
		else:
			exponent = np.tile(particles[i,:],(code.N,1))-np.tile(thets,(nparticles,1)).T
			mus = np.tile([n.mu for n in code.neurons],(nparticles,1)).T
			rs = np.exp(-0.5*exponent**2/code.alpha**2)*code.neurons[0].phi
			rs = rs*mus
			rt = np.sum(rs,axis=0)
			if rt*dt>1.0:
				print "WTF??????"
			weights[i,:] = weights[i-1,:]*(1.0-rt*dt)
			weights[i,:] = weights[i,:]/np.sum(weights[i,:])
		if np.sum(weights[i,:]**2)>essthreshold:
			particles[i,:] = particles[i,choice(weights[i,:],shape=particles[i,:].shape,randomstate=randomstate)]
			weights[i,:] = 1.0/nparticles	
	
	(m,st) = weighted_avg_and_std(particles,weights,nparticles,axis=1,testf=testf)
	mse = np.average((m-map(testf,s))**2)
	return [m,st,sps,s,mse,particles,weights]


def weighted_avg_and_std(values, ws,nparticles,axis=None,testf = (lambda x: x)):
	average = np.repeat(np.array([np.average(map(testf,values),weights=ws,axis=1)]).T,nparticles,axis=1)
	variance = np.sum(ws*(map(testf,values)-average)**2,axis=1)  # Fast and numerically precise
	return (average[:,0], np.sqrt(variance))
