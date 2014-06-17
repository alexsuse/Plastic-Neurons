#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python
"""Particle filtering for nonlinear dynamic systems observed through adaptive poisson neurons"""
import matplotlib
matplotlib.use('Agg')
import particlefilter as pf
import gaussianenv as ge
import poissonneuron as pn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#parameter definitions

dt = 0.001
phi = 1.0
zeta = 1.0
eta = 3.0
gamma = 1.0
alpha = 0.5
timewindow = 5000
dm = 0.0
nparticles = 200

tau = 1.0
plotting = True
gaussian = True

#env is the "environment", that is, the true process to which we don't have access

env_rng = np.random.mtrand.RandomState()

env = ge.BistableEnv(1.5,gamma,eta,1,randomstate=env_rng)
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
    [mg,varg,spsg,sg,mseg] = pf.gaussian_filter(code,env,timewindow=timewindow,dt=dt,mode = 'v',dense=True)
    stg = np.sqrt(varg)
    print stg.shape, varg.shape

env_rng.seed(12345)
code_rng.seed(67890)

env.reset(np.array([0.0]))
code.reset()
#[mp,varp,spsp,sp,msep,parts,ws] = pf.particle_filter(code,env,timewindow=timewindow,dt=dt,nparticles=nparticles,mode = 'v',testf = (lambda x:x))
[s,msep,spikecount,m,st,sptrain,sptimes] = pf.fast_particle_filter(code,env,timewindow=timewindow,dt=dt,nparticles=nparticles,mode = 'v',testf = (lambda x:x),dense=True)
if gaussian:
    print "MSE of gaussian filter %f"% mseg
print "MSE of particle filter %f"% msep


times = np.arange(0.0,dt*timewindow,dt)
plt.figure()

ax1 = plt.gcf().add_subplot(1,1,1)
ax1.plot(times,s,'r',label = 'True Sate')

#m = np.average(particles,weights=weights,axis=1)
#st = np.std(particles,weights=weights,axis=1)
#ext = (0.0,dt*timewindow,code.neurons[-1].theta,code.neurons[0].theta)
#plt.imshow(rates.T,extent=ext,cmap = cm.gist_yarg,aspect = 'auto',interpolation ='nearest')
thetas = [code.neurons[i].theta for i in sptrain]
ax1.plot(times[sptimes],thetas,'yo',label='Observed Spikes')
ax1.plot(times,m,'b',label='Posterior Mean')
ax1.plot(times,m-st,'gray',times,m+st,'gray')
#ax2 = plt.gcf().add_subplot(1,2,2)
#ax2.plot(times,s)
plt.xlabel('Time (in seconds)')
plt.ylabel('Space (in cm)')
plt.legend()
plt.title('State Estimation in a Diffusion System')
plt.savefig('filtering.png',dpi=600)



if plotting:
    
    matplotlib.rcParams['font.size']=10
    
    plt.close()    
    plt.figure()
    if gaussian:
        ax1 = plt.gcf().add_subplot(2,1,1)
    times = np.arange(0.0,dt*timewindow,dt)
    if gaussian:    
        ax1.plot(times,sg,'r',label='True State')
        if sum(sum(spsg)) !=0:
            (ts,neurs) = np.where(spsg == 1)
            spiketimes = times[ts]
            thetas = [code.neurons[i].theta for i in neurs]
        else:
            spiketimes = []
            thetas = []
        
        ax1.plot(spiketimes,thetas,'yo',label='Observed Spikes')
        ax1.plot(times,mg,'b',label='Posterior Mean')
        print (mg+stg).shape, (mg-stg).shape, times.shape
        ax1.fill_between(times,mg-stg,mg+stg,facecolor='gray')
        ax1.set_title('Gaussian Filter')
        ax1.set_ylabel('Position [cm] (Preferred Stimulus)')
        ax1.legend()
    
    if gaussian:
        ax2 = plt.gcf().add_subplot(2,1,2)
    else:
        ax2 = plt.gcf().add_subplot(1,1,1)
    
    thetas = [code.neurons[i].theta for i in sptrain]
    ax2.plot(times,s,'r',label = 'True Sate')
    ax2.plot(times[sptimes],thetas,'yo',label='Observed Spikes')
    ax2.plot(times,m,'b',label='Posterior Mean')
    ax2.fill_between(times,m-st,m+st,facecolor='gray')
    ax2.set_ylabel('Position [cm] (Preferred Stimulus)')
    ax2.set_xlabel('Time')
    ax2.legend()
    ax2.set_title('Particle Filter')

plt.show()    
plt.savefig('filtering_both.png')
