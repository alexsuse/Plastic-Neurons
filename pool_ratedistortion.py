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

gamma = 1.0
eta = 1.0
alpha = 1.0
delta = 0.1
tau = 1.0
thetas = np.arange(-4.0,4.0,0.2)

dt = 0.001
zeta = 1.0
timewindow = 1000000
nparticles = 200
alpha = 1.0

def runPF(params):
	[phi,delta] = params
	env_rng = np.random.mtrand.RandomState()
	
	env = ge.GaussianEnv(gamma=gamma,eta=eta,zeta=zeta,x0=0.0,y0=.0,L=1.0,N=1,
                         order=1,sigma=0.1,Lx=1.0,Ly=1.0,randomstate=env_rng)
	env.reset(np.array([0.0]))
	
	#code is the population of neurons, plastic poisson neurons	
	code_rng = np.random.mtrand.RandomState()
	code = pn.PoissonPlasticCode(A=alpha,phi=phi,tau=tau,thetas=thetas,dm=delta,
                                 randomstate=code_rng,alpha=alpha)
	
	# s is the stimulus, sps holds the spikes, rates the rates of each neuron
    # and particles give the position of the particles
	# weights gives the weights associated with each particle
	
	env_rng.seed(12345)
	code_rng.seed(67890)
	
	env.reset(np.array([0.0]))
	code.reset()
	
	[mmse,spikecount] = pf.mse_particle_filter(code,env,timewindow=timewindow,dt=dt,
                                               nparticles=nparticles,mode = 'Silent')
	print "ping "+str(phi)+" "+str(delta)+" "+str(mmse)+" "+str(spikecount)
	return [phi,delta,mmse, spikecount]

if __name__=='__main__':
#parameters for running
	deltas = np.arange(0.0,0.6,0.1)
	phis = np.arange(0.0,100.0,0.5)

#pool initalization
	ncpus = mp.cpu_count()
	pool = Pool(processes= ncpus)

#parmeters
	params = [[p,d] for p in phis for d in deltas]

#DO IT, DO IT NOW!
	outp = pool.map(runPF,params)

#Post processing
	os.system("""echo "post-processing now..."|mail -s "Simulation" alexsusemihl@gmail.com""")

#initialize dictionaries and arrays
	mmsedic = {}
	spikedic = {}
	nphis = phis.size
	ndeltas = deltas.size
	mmse = np.zeros((nphis,ndeltas))
	spcount = np.zeros((nphis,ndeltas))

#parse pool output
	for o in outp:
		[phi,delta,rest,spikec] = o
		mmsedic[(phi,delta)] = rest
		spikedic[(phi,delta)] = spikec

#store it in arrays
	for i,p in enumerate(phis):
		for j,t in enumerate(deltas):
			mmse[i,j] = mmsedic[(p,t)]
			spcount[i,j] = spikedic[(p,t)]

#print it, pickle it, pack it, technologic technologic technologic
	if len(sys.argv)>1:
		filename = sys.argv[1]
	else:
		filename = "../data/pickle_alphas_1"
	fi= open(filename,'w')
	pic.dump([mmse,spcount,phis,deltas],fi)
	os.system("""echo "simulation is ready, dude!"|mail -s "Simulation" alexsusemihl@gmail.com""")
