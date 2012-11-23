import numpy as np

class lateralInhibitionCode(object):
	def __init__(self,alpha,phi,tau,zeta,thetas):
		assert alpha > 0
		assert tau > 0
		assert phi > 0
		assert zeta > 0
		self.alpha = alpha
		self.phi = phi
		self.tau = tau
		self.zeta = zeta
		self.thetas = thetas
		self.kappas = np.ones(shape = self.thetas.shape)
		self.rng = np.random.mtrand.RandomState(123456)

	def inhibitionKernel(self,i):
		return np.exp(-0.5*(self.thetas-self.thetas[i])**2/self.zeta)

	def spike(self,X,dt):
		rates = self.kappas*np.exp(-0.5*(self.thetas-X)**2/self.alpha**2)*self.phi
		totalRate = dt*np.sum(rates)
		if self.rng.rand()<totalRate:
			#choose which neuron spiked
			spiker = choice(rates,randomstate = self.rng)
		else:
			spiker = None
		self.updateKappas(spiker,dt)
		return spiker

	def updateKappas(self,spiker,dt):
		if spiker:
			self.kappas = np.array(map(lambda x : np.max([x,0]),self.kappas-self.inhibitionKernel(spiker)))
		else:
			self.kappas = self.kappas*(1.0-dt/self.tau)+dt/self.tau

	def reset(self):
		self.kappas = np.ones(shape = self.kappas.shape)

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

if __name__=="__main__":
	inhibCode = lateralInhibitionCode(alpha=0.5,phi=1.0,tau=1.0,thetas = np.arange(-4.0,4.0,0.1),zeta = 0.9)
	dt = 0.001
	xs = np.arange(-2.0,2.0,0.2)
	bins = np.arange(-0.5,len(inhibCode.thetas)+0.5,1.0)
	results = {}
	histograms = {}
	for x in xs:
		inhibCode.reset()
		results[x]=[]
		for i in range(10000):
			spike = inhibCode.spike(x,dt)
			if spike:
				results[x].append(spike)
		results[x] =  np.array(results[x]).ravel()
		histograms[x] = np.histogram(results[x],bins=bins)

