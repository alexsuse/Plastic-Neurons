#!/usr/bin/python
import gaussianenv as ge
import particlefilter as pf
import poissonneuron as pn
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import cPickle as pic
import os
import sys

dt = 0.001
phi = 1.2
zeta = 1.0
eta = 1.8
gamma = 1.2
timewindow = 50000
dm = 0.0
alpha = 1.0
tau = 1.0
nparticles = 300
    
def runGauss(dtheta):
    env_rng = np.random.mtrand.RandomState()
    
    env = ge.GaussianEnv(gamma=gamma,eta=eta,zeta=zeta,
                         x0=0.0,y0=.0,L=1.0,N=1,order=1,
                         sigma=0.1,Lx=1.0,Ly=1.0,randomstate=env_rng)
    env.reset(np.array([0.0]))
    
    #code is the population of neurons, plastic poisson neurons    
    code_rng = np.random.mtrand.RandomState()
    code = pn.PoissonPlasticCode(A=alpha,phi=phi,tau=tau,
                                 thetas=np.arange(-10.0,10.0,dtheta),
                                 dm=dm,randomstate=code_rng,alpha=alpha)
    
    #s is the stimulus, sps holds the spikes, rates the rates of 
    # each neuron and particles give the position of the particles
    # weights gives the weights associated with each particle
    
    env_rng.seed(12345)
    code_rng.seed(67890)
    
    env.reset(np.array([0.0]))
    code.reset()
    
    results = pf.gaussian_filter(code,env,timewindow=timewindow,dt=dt,
                                 mode = 'Silent', dense = True)

    presults = pf.fast_particle_filter(code,env,timewindow=timewindow,dt=dt,
                                  nparticles=nparticles, mode='Silent',
                                  dense=False)
                    
    print "ping"    
    return [dtheta,results,presults]

if __name__=='__main__':
    ncpus = mp.cpu_count()
    pool = Pool(processes= ncpus)
    params = np.arange(0.01,2.0,0.05)
    outp = pool.map(runGauss,params)
    os.system("""echo "post-processing now..."|mail -s "Simulation" alexsusemihl@gmail.com""")
    outpickle = {}
    mmse = np.zeros((params.size))
    for o in outp:
        [nparticle,rest] = o
        outpickle[nparticle] = rest
    for i,a in enumerate(nparticles):
            mmse[i] = outpickle[a]
    if len(sys.argv)>1:
        filename = sys.argv[1]
    else:
        filename = "pickle_alphas_1"
    fi= open(filename,'w')
    pic.dump([mmse,nparticles],fi)
    os.system("""echo "simulation is ready, dude!"|mail -s "Simulation" alexsusemihl@gmail.com""")
