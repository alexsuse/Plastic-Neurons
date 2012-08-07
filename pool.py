import gaussianenv as ge
import particlefilter as pf
import poissonneuron as pn
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import cPickle as pic

dt = 0.001
phi = 1.2
alpha = 0.2
zeta = 1.0
eta = 1.8
gamma = 1.2
timewindow = 100
dm = 0.2
tau = 0.5
nparticles = 20
	
def runPF(alpha):
	
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
	
	results = pf.particle_filter(code,env,timewindow=timewindow,dt=dt,nparticles=nparticles,mode = 'Silent')
	
	return (alpha,results)

if __name__=='__main__':
	inp = np.arange(0.001,4.0,0.05)
	
	ncpus = mp.cpu_count()
	pool = Pool(processes= ncpus)
	outp = pool.map(runPF,inp)
	outpickle = {}
	for o in outp:
		[alpha,rest] = o
		outpickle[alpha] = rest
	fi= open('pickledump','w')
	pic.dump(outpickle,fi)
