#/!/Library/Frameworks/Python.framework/Versions/2.7/bin/python
"""Particle filtering for nonlinear dynamic systems observed through adaptive poisson neurons"""
from prettyplotlib import plt
import prettyplotlib as ppl
import particlefilter as pf
import gaussianenv as ge
import poissonneuron as pn
import numpy as np
import cPickle as pic
import sys
import multiprocessing as mp

#parameter definitions

#env is the "environment", that is, the true process to which we don't have access

#s is the stimulus, sps holds the spikes, rates the rates of each neuron
#and particles give the position of the particles
#weights gives the weights associated with each particle

dthetas = np.arange(0.1,2.0,0.1)

sparse_eps = np.zeros((dthetas.size,))
dense_eps = np.zeros((dthetas.size,))
particle_eps = np.zeros((dthetas.size,))

def run_filters(arg):
    i,dtheta = arg
    print arg
    dt = 0.001
    phi = 1.0
    zeta = 1.0
    eta = 1.0
    gamma = 1.0
    alpha = 0.5
    timewindow = 500000
    dm = 0.0
    tau = 1.0
    nparticles = 1000

    env_rng = np.random.mtrand.RandomState()

    env = ge.GaussianEnv(gamma=gamma,eta=eta,zeta=zeta,x0=0.0,
                         y0=.0,L=1.0,N=1,order=1,sigma=0.1,
                         Lx=1.0,Ly=1.0,randomstate=env_rng)
    env.reset(np.array([0.0]))

    #code is the population of neurons, plastic poisson neurons

    code_rng = np.random.mtrand.RandomState()
    
    code = pn.PoissonPlasticCode(A=alpha,phi=phi/2,tau=tau,
                                 thetas=np.arange(-20.0,20.0,dtheta),
                                 dm=dm,randomstate=code_rng)
    
    env_rng.seed(12345)
    code_rng.seed(67890)
    env.reset(np.array([0.0]))
    code.reset()
    
    [densem,densevar,spsg,sg,dense_mse] = pf.gaussian_filter(code,env,timewindow=timewindow,dt=dt,
                                                            dense=True)
    
    env_rng.seed(12345)
    code_rng.seed(67890)
    env.reset(np.array([0.0]))
    code.reset()
    
    [sparsem,sparsevar,spsg,sg,sparse_mse] = pf.gaussian_filter(code,env,timewindow=timewindow,dt=dt,
                                                                dense=False)
    
    env_rng.seed(12345)
    code_rng.seed(67890)
    env.reset(np.array([0.0]))
    code.reset()
    
    [msep,ws] = pf.mse_particle_filter(code, env, timewindow=timewindow,
                                                         dt=dt, nparticles=nparticles,
                                                         testf=(lambda x:x))

    return i,dense_mse,sparse_mse,msep

try:
    filename = sys.argv[1]
except:
    filename = 'dense_to_sparse.pik'

try:
    dic = pic.load( open(filename,'r'))
    sparse_eps = dic['sparse_eps']
    dense_eps = dic['dense_eps']
    particle_eps = dic['particle_eps']
    dthetas = dic['dthetas']
    print "ALL GOOD"

except:

    p = mp.Pool(mp.cpu_count())
    results = p.map(run_filters,zip(range(dthetas.size),dthetas))

    for r in results:
        i,dense,sparse,part = r
        dense_eps[i] = dens
        sparse_eps[i] = sparse
        particle_eps[i] = part

    with open(filename,"wb") as f:
        pic.dump({'dense_eps':dense_eps,
                  'sparse_eps':sparse_eps,
                  'particle_eps':particle_eps,
                  'dthetas':dthetas},
                  f)

if plotting:
    
    #matplotlib.rcParams['font.size']=10
    fig, ax = ppl.subplots(1,figsize=(5,5))

    ppl.plot(dthetas,sparse_eps,ax=ax,label='Full ADF')
    ppl.plot(dthetas,dense_eps,ax=ax,label='Dense ADF')
    ppl.plot(dthetas,particle_eps,ax=ax,label='Particle Filter')

    ppl.legend(ax)

    plt.show()
