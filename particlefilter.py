#!/usr/bin/python
"""Particle filtering for nonlinear dynamic systems observed through adaptive poisson neurons
"""
import numpy as np

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


#evaluates the mse of a particle filter for function f
def fast_particle_filter(code,env,timewindow=20000,dt=0.001,nparticles=20,mode='Silent',randomstate=np.random,testf = (lambda x: x)):

	mse=0.0
	sps = np.zeros((code.N))
	particles = np.zeros((nparticles))
	weights = np.ones((nparticles))/nparticles
	spcount = 0.0
	sptrain = []
	sptimes = []
	m = np.zeros(timewindow)
	st = np.zeros(timewindow)
	s = np.zeros(timewindow)

	thets = np.array([n.theta for n in code.neurons])
	essthreshold= 2.0/nparticles
	eta = env.geteta()	
	gamma = env.getgamma()
	
	particles = randomstate.normal(env.getstate(),eta**2/(2*gamma),particles.shape)
	olda = ""
	for i in range(timewindow):
		stim = env.samplestep(dt,N=1).ravel()
		s[i] = stim
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
		exponent = stim-thets
		grates = np.exp(-0.5*exponent**2/code.alpha**2)
		[temp1,_] = code.spikes(stim,dt,grates=grates)
		sps = temp1
		particles = particles+dt*env.drift(particles).ravel()+np.sqrt(dt)*randomstate.normal(0.0,eta,nparticles)
		a = np.where(sps==1)[0]
		if a:
			spcount +=1.0
			sptrain.append(a)
			sptimes.append(i)
			liks = code.neurons[a[0]].likelihood(particles)
			weights = weights*liks
			if np.sum(weights)==0.0:
				print "DANGER, DANGER"
				print np.sum(weights)
				print liks
				print weights
				weights[:] = 1.0/nparticles
			weights = weights/np.sum(weights)
		else:
			exponent = np.tile(particles,(code.N,1))-np.tile(thets,(nparticles,1)).T
			mus = np.tile([n.mu for n in code.neurons],(nparticles,1)).T
			rs = np.exp(-0.5*exponent**2/code.alpha**2)*code.neurons[0].phi
			rs = rs*mus
			rt = np.sum(rs,axis=0)
			weights = weights*(1.0-rt*dt)
			weights = weights/np.sum(weights)
		if np.sum(weights**2)>essthreshold:
			particles = particles[choice(weights,shape=particles.shape,randomstate=randomstate)]
			weights[:] = 1.0/nparticles	
		(m[i],st[i]) = weighted_avg_and_std(particles,weights,nparticles,testf=testf)
		mse += (m[i]-testf(stim))**2
	frate = spcount/(dt*timewindow)
	mse = mse/float(timewindow)
	return [s,mse,frate,m,st,sptrain,sptimes]

#evaluates the mse of a particle filter for function f
def mse_particle_filter(code,env,timewindow=20000,dt=0.001,nparticles=20,mode='Silent',randomstate=np.random,testf = (lambda x: x)):

	mse=0.0
	sps = np.zeros((code.N))
	particles = np.zeros((nparticles))
	weights = np.ones((nparticles))/nparticles
	spcount = 0.0
	
	thets = np.array([n.theta for n in code.neurons])
	essthreshold= 2.0/nparticles
	eta = env.geteta()	
	gamma = env.getgamma()
	
	particles = randomstate.normal(env.getstate(),eta**2/(2*gamma),particles.shape)
	olda = ""
	for i in range(timewindow):
		stim = env.samplestep(dt,N=1).ravel()
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
		exponent = stim-thets
		grates = np.exp(-0.5*exponent**2/code.alpha**2)
		[temp1,_] = code.spikes(stim,dt,grates=grates)
		sps = temp1
		particles = particles+dt*env.drift(particles).ravel()+np.sqrt(dt)*randomstate.normal(0.0,eta,nparticles)
		a = np.where(sps==1)[0]
		if a:
			spcount +=1.0
			liks = code.neurons[a[0]].likelihood(particles)
			weights = weights*liks
			if np.sum(weights)==0.0:
				print "DANGER, DANGER"
				print "total weight", np.sum(weights)
				print "likelihoods", liks
				print "weights", weights
				print "particles",particles
				print "spiker", a[0]
				print "mu", code.neurons[a[0]].mu
				print "theta",code.neurons[a[0]].theta
				weights[:] = 1.0/nparticles
			weights = weights/np.sum(weights)
		else:
			exponent = np.tile(particles,(code.N,1))-np.tile(thets,(nparticles,1)).T
			mus = np.tile([n.mu for n in code.neurons],(nparticles,1)).T
			rs = np.exp(-0.5*exponent**2/code.alpha**2)*code.neurons[0].phi
			rs = rs*mus
			rt = np.sum(rs,axis=0)
			weights = weights*(1.0-rt*dt)
			weights = weights/np.sum(weights)
		if np.sum(weights**2)>essthreshold:
			particles = particles[choice(weights,shape=particles.shape,randomstate=randomstate)]
			weights[:] = 1.0/nparticles	
		m = np.sum(testf(particles)*weights)
		mse += (m-testf(stim))**2
	frate = spcount/(dt*timewindow)
	mse = mse/float(timewindow)
	return [mse,frate]


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
			weights[i,:] = weights[i-1,:]*(1.0-rt*dt)
			weights[i,:] = weights[i,:]/np.sum(weights[i,:])
		if np.sum(weights[i,:]**2)>essthreshold:
			particles[i,:] = particles[i,choice(weights[i,:],shape=particles[i,:].shape,randomstate=randomstate)]
			weights[i,:] = 1.0/nparticles	

	(m,st) = weighted_avg_and_std(particles,weights,nparticles,ax=1,testf=testf)
	mse = np.average((m-map(testf,s))**2)
	return [m,st,sps,s,mse,particles,weights]


def weighted_avg_and_std(values, ws,nparticles,ax=0,testf = (lambda x: x)):
	average = np.average(map(testf,values),weights=ws,axis=ax)
	variance = np.average(np.array(map(testf,values))**2,weights=ws,axis=ax)-average**2
	#average = np.repeat(np.array([np.average(map(testf,values),weights=ws,axis=axis)]).T,nparticles,axis=axis)
	#else:
	#	average = np.average(map(testf,values),weights=ws)
	#	variance = np.sum(ws*(map(testf,values)-average)**2)
		
	#variance = np.sum(ws*(map(testf,values)-average)**2,axis=axis)  # Fast and numerically precise
	return (average, np.sqrt(variance))
