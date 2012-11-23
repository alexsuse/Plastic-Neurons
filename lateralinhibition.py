import numpy as np
import matplotlib.pyplot as plt

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
		return np.exp(self.zeta*(-1.0+np.cos(self.thetas-self.thetas[i])))

	def spike(self,X,dt):
		rates = self.kappas*np.exp(self.alpha*(-1.0+np.cos(self.thetas-X)))*self.phi
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
	nneurons = 10
	neurons = np.arange(-np.pi,np.pi,2*np.pi/nneurons)
	alpha=0.9
	phi = 100.0
	tau = 1.0
	zeta = 0.4
	inhibCode = lateralInhibitionCode(alpha=alpha,phi=phi,tau=tau,thetas = neurons,zeta = zeta)
	dt = 0.001
	Ntrials = 1000000
	xs = np.arange(-np.pi,np.pi,0.5*np.pi/nneurons)
	ps = np.ones(shape = xs.shape)
	ps[nneurons/2] = ps[nneurons/2] + 0.8
	ps[0] = ps[0]+0.4
	ps = ps/np.sum(ps)
	bins = np.arange(-0.5,len(inhibCode.thetas)+0.5,1.0)
	results = {}
	histograms = {}
	for n in neurons:
		histograms[n] = []
	for x in xs:
		results[x]=[]
	for i in range(Ntrials):
		print i/np.float(Ntrials)
		x = choice(ps,a=xs)[0]
		spike = inhibCode.spike(x,dt)
		if spike!=None:
			spike= spike[0]
			results[x].append(spike)
			histograms[neurons[spike]].append(x)
	for x in xs:
		results[x] =  np.array(results[x]).ravel()
	for x in neurons:
		histograms[x] = np.array(histograms[x]).ravel()

	hists = np.zeros(shape=(nneurons,2*nneurons))
	bins = np.arange(-np.pi,np.pi+0.01,np.pi/nneurons)
	for i,x in enumerate(neurons):
		print len(histograms[x])
		hists[i,:] = np.histogram(histograms[x],bins=bins,normed=True)[0]

	for i,x in enumerate(neurons):
		plt.plot(bins[:-1],hists[i,:])
		plt.plot(bins[:-1],np.exp(alpha*(-1.0+np.cos(bins[:-1]-x))))
		plt.show()

	#plt.imshow(hists,interpolation='nearest')
	#plt.show()
	#histograms[x] = np.histogram(results[x],bins=bins)

