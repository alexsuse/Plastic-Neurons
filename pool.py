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
phi = 1.0
zeta = 1.0
eta = 1.0
gamma = 1.0
timewindow = 100000
dm = 0.5
nparticles = 2000

def runPF(params):
	[alpha,tau] = params
	env_rng = np.random.mtrand.RandomState()
	
	#env = ge.GaussianEnv(gamma=gamma,eta=eta,zeta=zeta,x0=0.0,y0=.0,L=1.0,N=1,order=1,sigma=0.1,Lx=1.0,Ly=1.0,randomstate=env_rng)
	env = ge.GaussianEnv(gamma=gamma,eta=eta,zeta=zeta,x0=0.0,y0=.0,L=1.0,N=1,order=1,sigma=0.1,Lx=1.0,Ly=1.0,randomstate=env_rng)
	env.reset(np.array([0.0]))
	
	#code is the population of neurons, plastic poisson neurons	
	code_rng = np.random.mtrand.RandomState()
	code = pn.PoissonPlasticCode(A=alpha,phi=phi,tau=tau,thetas=np.arange(-4.0,4.0,0.2),dm=dm,randomstate=code_rng,alpha=alpha)
	
	#s is the stimulus, sps holds the spikes, rates the rates of each neuron and particles give the position of the particles
	#weights gives the weights associated with each particle
	
	env_rng.seed(12345)
	code_rng.seed(67890)
	
	env.reset(np.array([0.0]))
	code.reset()
	
	[mmse,spikecount] = pf.mse_particle_filter(code,env,timewindow=timewindow,dt=dt,nparticles=nparticles,mode = 'Silent',testf = lambda x : x)
	print "ping "+str(alpha)+" "+str(tau)+" "+str(mmse)+" "+str(spikecount)
	return [alpha,tau,mmse, spikecount]

if __name__=='__main__':
#define parameter ranges
	alpha = np.arange(0.001,4.0,0.2)
	taus = np.arange(0.001,4.0,0.8)

#initialize multiprocesning pool
	ncpus = mp.cpu_count()
	pool = Pool(processes= ncpus)

#define parameter list
	params = [[a,t] for a in alpha for t in taus]

#run particle filter on parameters through pool
	outp = pool.map(runPF,params)

#post-processing
	os.system("""echo "post-processing now..."|mail -s "Simulation" alexsusemihl@gmail.com""")

#initialize dictionaries for parsing
	mmsedic = {}
	spikedic = {}

#allocate output arrays
	nalphas = alpha.size
	ntaus = taus.size
	mmse = np.zeros((nalphas,ntaus))
	spcount = np.zeros((nalphas,ntaus))

#parse output into dict
	for o in outp:
		[al,tau,rest,spikec] = o
		mmsedic[(al,tau)] = rest
		spikedic[(al,tau)] = spikec

#get data from dict into  array
	for i,a in enumerate(alpha):
		for j,t in enumerate(taus):
			mmse[i,j] = mmsedic[(a,t)]
			spcount[i,j] = spikedic[(a,t)]

#find output file
	if len(sys.argv)>1:
		filename = sys.argv[1]
	else:
		filename = "../data/pickle_alphas_1"

#dump and go
	np.savez(file = filename, eps = mmse, alphas = alpha, taus = taus, delta = dm)
	fi= open(filename+'.pik','wb')
	pic.dump([mmse,spcount,alpha,taus],fi)
	os.system("""echo "simulation is ready, dude!"|mail -s "Simulation" alexsusemihl@gmail.com""")
