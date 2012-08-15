import numpy as np
import random

class PoissonPlasticNeuron(object):
	"""Implements a poisson neuron with a gaussian tuning function"""
	def __init__(self,theta,A,phi,tau,N=None,dm=0.05,randomstate = None,alpha=1.0):
		"""Generates the neuron object,
		X are the coordinates of the stimulus points
		A is the correlation of the gaussian tuning function
		phi is the maximal firing rate
		inva the inverse of a is calculated once for reuse"""
		if randomstate==None:
			self.rng = np.random
		else:
			self.rng=randomstate	
		self.A=A
		if N == None:
			N = np.size(theta)
		else:
			self.N = N
		self.inva = np.array(np.matrix(A).I)
		self.phi=phi
		self.theta = theta
		self.mu = 1.0
		self.dm = dm
		self.tau = tau
		self.alpha=alpha
	def getmu(self):
		return self.mu

	def rate(self,S):
		"""Gives the rate of the poisson spiking process given the stimulus S"""
		#S = np.reshape(stim,self.N*self.N)
		#exponent = np.dot(S.transpose()-self.theta.transpose(),np.dot(self.inva,S-self.theta))
		exponent = (S-self.theta)**2/self.alpha**2
		return self.phi*self.mu*np.exp(-0.5*exponent).ravel().ravel()
	def spike(self,S,dt,rate = None):
		"""Generates a spike with probability rate*dt"""
		if rate == None:
			r = dt*self.rate(S)
		else:
			r = rate*dt
		if self.rng.uniform()<r:
			self.oldmu = self.mu+(1.0-self.mu)*dt/self.tau
			self.mu = self.mu - self.dm
<<<<<<< HEAD
			self.mu = self.mu if self.mu > 0.00 else 0.0001
=======
			self.mu = self.mu if self.mu > 0.00 else 0.000
>>>>>>> old-state
			return [1,r]
		self.mu = self.mu+(1.0-self.mu)*dt/self.tau
		return [0,r]
	def likelihood(self,x):
		liks = np.zeros_like(x)
		exponent = (x-self.theta)**2/self.alpha**2
		liks = np.exp(-0.5*exponent)*self.phi
		return liks
	def resetmu(self,x=None):
		if x==None:
			tem = self.mu
			self.mu = self.oldmu
			self.oldmu = self.mu
		else:
			self.mu= 1.0

def choice(p,a=None,shape=(1,),randomstate = None):
	"""chooses an element from a with probabilities p. Can return arbitrarily shaped-samples through the shape argument.
	p needs not be normalized, as this is checked for."""
	if randomstate==None:
		x = np.random.uniform(size=shape)	
	else:
		x = randomstate.uniform(size=shape)
	x = np.random.uniform(size=shape)
	cump = np.cumsum(p)
	if cump[-1]!=1:
		cump=cump/cump[-1]
	idxs = np.searchsorted(cump,x)
	if a==None:
		return idxs
	else:
		return a[idxs]


class PoissonPlasticCode(object):
	def __init__(self,thetas=np.arange(-2.0,2.0,0.1),A = None, phi=1,N=40,alpha=1.0,tau=0.4,dm=0.05,randomstate = None):
		self.N = np.size(thetas)
		self.A = A
		self.alpha = alpha
		if randomstate == None:
			self.rng = np.random
		else:
			self.rng = randomstate
		if A == None:
			self.A= alpha*np.eye(np.size(thetas[0]))
		self.neurons = []
		for theta in thetas:
			self.neurons.append(PoissonPlasticNeuron(theta,self.A,phi,tau,dm=dm,randomstate=randomstate,alpha=alpha))
	def rates(self,stim):
		rs = []
		for n in self.neurons:
			rs.append(n.rate(stim))
		return rs
	def totalrate(self,stim):
		return np.sum(self.rates(stim))
	
	def reset(self):
		for i in self.neurons:
			i.resetmu(1.0)
	def spikes(self,stim,dt,grates=None):
		sps = np.zeros(self.N)
		if grates==None:
			rates = []
			for i,n in enumerate(self.neurons):
				[sps[i],r] = n.spike(stim,dt)
				rates.append(r)
		else:
			rates = []
			for i,n in enumerate(self.neurons):
				[sps[i],r] = n.spike(stim,dt,rate=n.phi*n.mu*grates[i])
				rates.append(r)
		if sum(sps) > 1:
			spikers = np.where(sps==1)[0]
			for i in spikers:
				self.neurons[i].resetmu()
			spikersrates = [rates[i] for i in spikers]
			spiker = choice(spikersrates,spikers)
			sps[:] = np.zeros_like(sps[:])
			sps[spiker]=1
			self.neurons[spiker].resetmu()
		return [sps,np.array(rates)]
"""	r = self.rates(stim)
		tot = np.sum(r)
		sps = np.zeros(self.N)
		if np.random.uniform() < tot*dt:
			neu = choice(p=r)
			sps[neu]=1
			self.neurons[neu].spike(stim,1/(dt*r[neu]))
"""
