import numpy as np
import random

class PoissonPlasticNeuron(object):
	"""Implements a poisson neuron with a gaussian tuning function"""
	def __init__(self,theta,A,phi,tau,N=None,dm=0.05):
		"""Generates the neuron object,
		X are the coordinates of the stimulus points
		A is the correlation of the gaussian tuning function
		phi is the maximal firing rate
		inva the inverse of a is calculated once for reuse"""
		
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
	def getmu(self):
		return self.mu

	def rate(self,S):
		"""Gives the rate of the poisson spiking process given the stimulus S"""
		#S = np.reshape(stim,self.N*self.N)
		exponent = np.dot(S.transpose()-self.theta.transpose(),np.dot(self.inva,S-self.theta))
		return self.phi*self.mu*np.exp(-0.5*exponent).ravel().ravel()
	def spike(self,S,dt):
		"""Generates a spike with probability rate*dt"""
		r = dt*self.rate(S)
		if np.random.uniform()<r:
			self.oldmu = self.mu+(1.0-self.mu)*dt/self.tau
			self.mu = self.mu - self.dm
			self.mu = self.mu if self.mu > 0.0 else 0.0
			return [1,r]
		self.mu = self.mu+(1.0-self.mu)*dt/self.tau
		return [0,r]
	def likelihood(self,x):
		liks = np.zeros_like(x)
		for (i,e) in enumerate(x):
			liks[i] = self.rate(e)
		return liks
	def resetmu(self):
		tem = self.mu
		self.mu = self.oldmu
		self.oldmu = self.mu
		

def choice(p,a=None,shape=(1,)):
	"""chooses an element from a with probabilities p. Can return arbitrarily shaped-samples through the shape argument.
	p needs not be normalized, as this is checked for."""
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
	def __init__(self,thetas=np.arange(-2.0,2.0,0.1),A = None, phi=1,N=40,alpha=1.0,tau=0.4,dm=0.05):
		self.N = np.size(thetas)
		self.A = A
		if A == None:
			self.A= alpha*np.eye(np.size(thetas[0]))
		self.neurons = []
		for theta in thetas:
			self.neurons.append(PoissonPlasticNeuron(theta,self.A,phi,tau,dm=dm))
	def rates(self,stim):
		rs = []
		for n in self.neurons:
			rs.append(n.rate(stim))
		return rs
	def totalrate(self,stim):
		return np.sum(self.rates(stim))
	def spikes(self,stim,dt):
		sps = np.zeros(self.N)
		rates = []
		for i,n in enumerate(self.neurons):
			[sps[i],r] = n.spike(stim,dt)
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
		return [sps,rates]
"""	r = self.rates(stim)
		tot = np.sum(r)
		sps = np.zeros(self.N)
		if np.random.uniform() < tot*dt:
			neu = choice(p=r)
			sps[neu]=1
			self.neurons[neu].spike(stim,1/(dt*r[neu]))
"""
