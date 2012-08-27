#!/usr/bin/python
"""calculate autocorrelation of spike trains for adaptive and non-adaptive neurons"""
import matplotlib
#matplotlib.use('Agg')
import gaussianenv as ge
import poissonneuron as pn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#parameter definitions

dt = 0.001
phi = 1.2
alpha = 0.9
zeta = 1.0
eta = 1.0
gamma = 1.2
timewindow = 100000
dm = 0.2
taus = np.arange(0.001,5.0,0.5)
plotting = True

#env is the "environment", that is, the true process to which we don't have access

env_rng = np.random.mtrand.RandomState()
env = ge.GaussianEnv(gamma=gamma,eta=eta,zeta=zeta,x0=0.0,y0=.0,L=1.0,N=1,order=1,sigma=0.1,Lx=1.0,Ly=1.0,randomstate=env_rng)
env.reset(np.array([0.0]))

acorr= {}

for tau in taus:
#code is the population of neurons, plastic poisson neurons
	code_rng = np.random.mtrand.RandomState()
	code = pn.PoissonPlasticCode(A=alpha,phi=phi,tau=tau,thetas=np.arange(-1.0,1.0,0.15),dm=dm,randomstate=code_rng)
	
	#s is the stimulus, sps holds the spikes, rates the rates of each neuron and particles give the position of the particles
	#weights gives the weights associated with each particle
	env_rng.seed(12345)
	code_rng.seed(67890)
	env.reset(np.array([0.0]))
	code.reset()
	
	s = env.samplestep(dt,N=timewindow).ravel()
	thetas = np.array([n.theta for n in code.neurons]) 
	exponents = np.tile(s,(code.N,1))-np.tile(thetas,(timewindow,1)).T
	grates = np.exp(-0.5*exponents**2/alpha**2).T
	
	sps = np.zeros((timewindow,code.N))
	
	for i in range(timewindow):
		[sps[i,:],_] = code.spikes(s[i],dt,grates=grates[i,:])
	
	spktimes = {}
	autocorr = np.zeros((code.N,200))
	counter = np.zeros(code.N)
	
	for i in range(code.N):
		spktimes[i] = np.where(sps[:,i]==1.0)[0]
		for t in spktimes[i]:
			print t
			if t>200:
				autocorr[i,:] += sps[t-200:t,i]
				counter[i] +=1
		if counter[i]!=0:
			autocorr[i,:]/=counter[i]
	print tau
	acorr[tau] = autocorr

if plotting:
	
	matplotlib.rcParams['font.size']=10
	plt.close()	
	plt.figure()
	ax1 = plt.gcf().add_subplot(1,1,1)
	times = np.arange(-dt*200,0.0,dt)
	for t in acorr.keys():	
		ax1.plot(times,acorr[t][13,:],label = 'tau = '+str(t))
	corr = np.exp(-2*gamma*np.abs(times))*eta**2/(2*gamma)
	ax1.plot(times,corr,label='stimulus')
	ax1.legend()
	plt.show()
