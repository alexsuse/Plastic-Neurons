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
x0 = 1.0
eta = 0.85
gamma = 1.0
timewindow = 3000000
dm = 0.2
nparticles = 200

f = lambda x : -1.0+2.0/(1.0+np.exp(-x))

def runPF(params):
	[alpha,tau] = params
	env_rng = np.random.mtrand.RandomState()
	
	#env = ge.GaussianEnv(gamma=gamma,eta=eta,zeta=zeta,x0=0.0,y0=.0,L=1.0,N=1,order=1,sigma=0.1,Lx=1.0,Ly=1.0,randomstate=env_rng)
	env = ge.BistableEnv(gamma=gamma,eta=eta,x0=x0,order=1,randomstate=env_rng)
	env.reset(np.array([0.0]))
	
	#code is the population of neurons, plastic poisson neurons	
	code_rng = np.random.mtrand.RandomState()
	code = pn.PoissonPlasticCode(A=alpha,phi=phi,tau=tau,thetas=np.arange(-3.0,3.0,0.15),dm=dm,randomstate=code_rng,alpha=alpha)
	
	#s is the stimulus, sps holds the spikes, rates the rates of each neuron and particles give the position of the particles
	#weights gives the weights associated with each particle
	
	env_rng.seed(12345)
	code_rng.seed(67890)
	
	env.reset(np.array([0.0]))
	code.reset()
	
	[mmse,spikecount] = pf.mse_particle_filter(code,env,timewindow=timewindow,dt=dt,nparticles=nparticles,mode = 'Silent',testf = f)
	print "ping "+str(alpha)+" "+str(tau)+" "+str(mmse)+" "+str(spikecount)
	return [alpha,tau,mmse, spikecount]

if __name__=='__main__':

#run parameters, alphas and taus
	alpha = np.arange(0.001,2.0,0.05)
	taus = np.arange(0.001,10.0,2.0)

#pool initialization
	ncpus = mp.cpu_count()
	pool = Pool(processes= ncpus)

#parameter list
	params = [[a,t] for a in alpha for t in taus]

#run pool with runPF and params
	outp = pool.map(runPF,params)
	
#POST-PROCESSING
	os.system("""echo "post-processing now..."|mail -s "Simulation" alexsusemihl@gmail.com""")

#mmses and spikecount lists and dicts
	mmsedic = {}
	spikedic = {}
	nalphas = alpha.size
	ntaus = taus.size
	mmse = np.zeros((nalphas,ntaus))
	spcount = np.zeros((nalphas,ntaus))

#parse output dictionary
	for o in outp:
		[al,tau,rest,spikec] = o
		mmsedic[(al,tau)] = rest
		spikedic[(al,tau)] = spikec

#store in mmse and spcount matrices
	for i,a in enumerate(alpha):
		for j,t in enumerate(taus):
			mmse[i,j] = mmsedic[(a,t)]
			spcount[i,j] = spikedic[(a,t)]

#if output is provided, pickle to output otherwise to default file
	if len(sys.argv)>1:
		filename = sys.argv[1]
	else:
		filename = "../data/pickle_alphas_1"
	fi= open(filename,'w')
	pic.dump([mmse,spcount,alpha,taus],fi)
	os.system("""echo "simulation is ready, dude!"|mail -s "Simulation" alexsusemihl@gmail.com""")
