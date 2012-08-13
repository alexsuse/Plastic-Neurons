#!/usr/bin/python
"""Particle filtering for nonlinear dynamic systems observed through adaptive poisson neurons"""
import particlefilter as pf
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
alpha = 0.2
zeta = 1.0
eta = 1.8
gamma = 1.2
timewindow = 2000
dm = 0.2
tau = 15.0
nparticles = 200
plotting = True
gaussian = False

#env is the "environment", that is, the true process to which we don't have access

env_rng = np.random.mtrand.RandomState()

env = ge.GaussianEnv(gamma=gamma,eta=eta,zeta=zeta,x0=0.0,y0=.0,L=1.0,N=1,order=1,sigma=0.1,Lx=1.0,Ly=1.0,randomstate=env_rng)
env.reset(np.array([0.0]))

#code is the population of neurons, plastic poisson neurons

code_rng = np.random.mtrand.RandomState()

code = pn.PoissonPlasticCode(A=alpha,phi=phi/2,tau=tau,thetas=np.arange(-20.0,20.0,0.15),dm=dm,randomstate=code_rng)

#s is the stimulus, sps holds the spikes, rates the rates of each neuron and particles give the position of the particles
#weights gives the weights associated with each particle

env_rng.seed(12345)
code_rng.seed(67890)

env.reset(np.array([0.0]))
code.reset()
if gaussian:
	[mg,varg,spsg,sg,mseg] = pf.gaussian_filter(code,env,timewindow=timewindow,dt=dt,mode = 'v')

env_rng.seed(12345)
code_rng.seed(67890)

env.reset(np.array([0.0]))
code.reset()

[mp,varp,spsp,sp,msep,parts,ws] = pf.particle_filter(code,env,timewindow=timewindow,dt=dt,nparticles=nparticles,mode = 'v',testf = (lambda x:1.0/(1.0+np.exp(-x))))

if gaussian:
	print "MSE of gaussian filter %f"% mseg
print "MSE of particle filter %f"% msep

if plotting:
	
	matplotlib.rcParams['font.size']=10
	
	plt.close()	
	plt.figure()
	ax1 = plt.gcf().add_subplot(2,1,1)
	times = np.arange(0.0,dt*timewindow,dt)
	if gaussian:	
		ax1.plot(times,sg,'r',label='Signal')
		if sum(sum(spsg)) !=0:
			(ts,neurs) = np.where(spsg == 1)
			spiketimes = times[ts]
			thetas = [code.neurons[i].theta for i in neurs]
		else:
			spiketimes = []
			thetas = []
		
		ax1.plot(spiketimes,thetas,'yo',label='Spike times')
		ax1.plot(times,mg,'b',label='Mean prediction')
		ax1.set_title('Gaussian Filter')
		ax1.set_ylabel('Signal space')
		ax1.legend()
	
	ax2 = plt.gcf().add_subplot(2,1,2)
	
	ax2.plot(times,sp,'r',label='Signal')
	if sum(sum(spsp)) !=0:
		(tsp,neursp) = np.where(spsp == 1)
		spiketimesp = times[tsp]
		thetasp = [code.neurons[i].theta for i in neursp]
	else:
		spiketimesp = []
		thetasp = []
	
	ax2.plot(spiketimesp,thetasp,'yo',label='Spike times')
	ax2.plot(times,mp,'b',label='Mean prediction')
	ax2.set_ylabel('Signal space')
	ax2.set_xlabel('Time')
	ax2.legend()
	ax2.set_title('Particle Filter')
	
	plt.savefig('filtering.png',dpi=150)
