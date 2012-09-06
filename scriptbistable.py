
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
alpha = 0.4
eta = 0.85
gamma = 2.0
timewindow = 50000
dm = 0.2
x0 = 1.0
tau = 1.0
nparticles = 100

#env is the "environment", that is, the true process to which we don't have access
env_rng = np.random.mtrand.RandomState()
env = ge.BistableEnv(gamma=gamma,eta=eta,x0=1.0,order=1,N=1,randomstate=env_rng)
env.reset(np.array([0.0]))

#code is the population of neurons, plastic poisson neurons

code_rng = np.random.mtrand.RandomState()
code = pn.PoissonPlasticCode(A=alpha,phi=phi,tau=tau,thetas=np.arange(-3.0,3.0,0.1),dm=dm,alpha=alpha,randomstate=code_rng)

env_rng.seed(12345)
code_rng.seed(67890)

#s is the stimulus, sps holds the spikes, rates the rates of each neuron and particles give the position of the particles
#weights gives the weights associated with each particle

f = lambda x : -1.0+2.0/(1.0+np.exp(-5*x))

[s,mse,frate,m,st,spiketrain,spiketimes] = pf.fast_particle_filter(code,env,dt=dt,timewindow=timewindow,nparticles=nparticles,mode='v',testf = f)

plt.close()	

times = np.arange(0.0,dt*timewindow,dt)
plt.figure()

ax1 = plt.gcf().add_subplot(1,1,1)
ax1.plot(times,map(f,s),'r',label = 'True Sate')

#m = np.average(particles,weights=weights,axis=1)
#st = np.std(particles,weights=weights,axis=1)
#ext = (0.0,dt*timewindow,code.neurons[-1].theta,code.neurons[0].theta)
#plt.imshow(rates.T,extent=ext,cmap = cm.gist_yarg,aspect = 'auto',interpolation ='nearest')
thetas = [code.neurons[i].theta for i in spiketrain]
ax1.plot(times[spiketimes],map(f,thetas),'yo',label='Observed Spikes')
ax1.plot(times,m,'b',label='Posterior Mean')
ax1.plot(times,m-st,'gray',times,m+st,'gray')
#ax2 = plt.gcf().add_subplot(1,2,2)
#ax2.plot(times,s)
plt.xlabel('Time (in seconds)')
plt.ylabel('Space (in cm)')
plt.legend()
plt.title('State Estimation in a Bistable System')
plt.savefig('filtering_bistable_sigmoid.png',dpi=300)


