#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python
"""Particle filtering for nonlinear dynamic systems observed through adaptive poisson neurons"""
import prettyplotlib as ppl
from prettyplotlib import plt
#import matplotlib
#matplotlib.use('Agg')
import particlefilter as pf
import gaussianenv as ge
import poissonneuron as pn
import numpy as np
#import matplotlib.pyplot as plt
from matplotlib import cm

#parameter definitions

dt = 0.001
phi = 0.4
zeta = 1.0
eta = 2.0
gamma = 2.0
alpha = 0.1
timewindow = 3000
dm = 0.0
nparticles = 500

tau = 1.0
plotting = True
gaussian = True

#env is the "environment", that is, the true process to which we don't have access

env_rng = np.random.mtrand.RandomState()

env = ge.BistableEnv(1.0,gamma,eta,1,randomstate=env_rng)
env.reset(np.array([0.0]))

#code is the population of neurons, plastic poisson neurons

code_rng = np.random.mtrand.RandomState()

code = pn.PoissonPlasticCode(A=alpha,phi=phi/2,tau=tau,thetas=np.arange(-5.0,5.0,0.05),dm=dm,randomstate=code_rng)

#s is the stimulus, sps holds the spikes, rates the rates of each neuron and particles give the position of the particles
#weights gives the weights associated with each particle
env_seed = np.random.randint(1e8)
code_seed = np.random.randint(1e8)
env_rng.seed(env_seed)
code_rng.seed(code_seed)

env.reset(np.array([0.0]))
code.reset()
if gaussian:
    [mg,varg,spsg,sg,mseg] = pf.gaussian_filter(code,env,timewindow=timewindow,dt=dt,mode = 'v',dense=True)
    stg = np.sqrt(varg)
    print stg.shape, varg.shape

env_rng.seed(env_seed)
code_rng.seed(code_seed)

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



if plotting:
    
    #matplotlib.rcParams['font.size']=10
    
    if gaussian:
        fig, (ax1,ax2) = ppl.subplots(1,2,figsize = (12,6))
    else:
        fig, ax2 = ppl.subplots(1)
    times = np.arange(0.0,dt*timewindow,dt)
    if gaussian:    
        if sum(sum(spsg)) !=0:
            (ts,neurs) = np.where(spsg == 1)
            spiketimes = times[ts]
            thetas = [code.neurons[i].theta for i in neurs]
        else:
            spiketimes = []
            thetas = []
        
        l4, = ax1.plot(times,sg,label='True State')
        l2, = ax1.plot(times,mg,label='Posterior Mean')
        l1, = ax1.plot(spiketimes,thetas,'o',label='Observed Spikes')
        l3 = ppl.fill_between(times,mg-stg,mg+stg,ax=ax1,alpha=0.2)
        c1 = l1.get_color()
        c2 = l2.get_color()
        c3 = l3.get_facecolor()
        c4 = l4.get_color()
        ax1.set_title('Gaussian Assumed Density Filter')
        ax1.set_ylabel('Position [cm] (Preferred Stimulus)')
        ax1.set_xlabel('Time [s]')
        ppl.legend(ax1).get_frame().set_alpha(0.6)
    
    
    thetas = [code.neurons[i].theta for i in sptrain]
    ax2.plot(times,s,color=c4,label = 'True Sate')
    ax2.plot(times,m,color=c2,label='Posterior Mean')
    ax2.plot(times[sptimes],thetas,'o',color=c1,label='Observed Spikes')
    ppl.fill_between(times,m-st,m+st,ax=ax2,facecolor=c3,alpha=0.2)
    ax2.set_ylabel('Position [cm] (Preferred Stimulus)')
    ax2.set_xlabel('Time [s]')
    ppl.legend(ax2).get_frame().set_alpha(0.6)
    
    ax2.set_title('Particle Filter')

    ax1.set_ylim([(m-st).min()-0.5,(m+st).max()+0.5])
    ax2.set_ylim([(m-st).min()-0.5,(m+st).max()+0.5])

#plt.show()    
plt.savefig('filtering_both.pdf')
plt.savefig('filtering_both.eps')
plt.savefig('filtering_both.png')
