import numpy as np

def factorial(n):
	temp = 1
	for i in range(n):
		temp*=(i+1)
	return temp

def binomial(n,k):
	return factorial(n)/(factorial(k)*factorial(n-k))


class GaussianEnv(object):
	"""Implements a gaussian process with correlation structure
	<S_i(t_k)S_j(t_l)> = exp(-(|x(i)-x(j)|/L)**zeta)*phi(|t_k-t_l|;eta,gamma,order)
	we take the points x to be regularly distributed on a grid on the square
	(x0,y0),(x0,y0+Ly),(x0+Lx,y0+Ly),(x0+Lx,y0) and phi is the matern kernel
	with \nu = order-1/2."""
	def __init__(self,zeta,gamma,eta,L,N,x0,y0,Lx,Ly,sigma,order):
		"""Constructer method, generates all the internal data
		xs and ys are the x and y coordinates of all the points.
		zeta, eta, gamma, L are the parameters of the kernel
		N is the number of subdivisions in the grid, sigma is added
		to the diagonal of the gram matrix to ensure positive definiteness
		and order is the order of the matern process
		k is the covariance matrix, and khalf its cholesky decomposition, used to sample"""
		self.order = order if order>0 else 1
		self.xs = np.arange(x0,x0+Lx,Lx/N)
		self.ys = np.arange(y0,y0+Ly,Ly/N)
		self.zeta = zeta
		self.gamma = gamma
		self.Gamma = self.getGamma()
		self.eta = eta
		self.H = self.getH()
		self.L = L
		self.N = N
		self.k = np.zeros((N*N,N*N))
		self.S = np.random.normal(0.0,1.0,(order,N*N))
		if N>1:
			for i in range(0,N*N):
				for j in range(0,N*N):
					(ix,iy) =divmod(i,N)
					(jx,jy) =divmod(j,N)
					dist = np.sqrt((self.xs[ix]-self.xs[jx])**2 + (self.ys[iy]-self.ys[jy])**2) 
					self.k[i,j] = np.exp(-(dist/L)**zeta)+sigma*(i==j)
			self.khalf = np.linalg.cholesky(self.k)
		else:
			self.khalf = np.array([1.0])
	def reset(self,x=None):
		if x:
			self.S = x
		else:
			self.S = np.random.normal(0.0,1.0,(self.order,self.N*self.N))

	def sample(self):
		"""Gets an independent sample from the spatial kernel"""
		s = np.random.normal(0.0,1.0,self.N*self.N)
		s = np.dot(self.khalf,s)	
		return np.reshape(s,(self.N,self.N))

	def drift(self,x=None):
		if x!=None:
			return -np.dot(self.Gamma,np.array([x]))
		else:
			return -np.dot(self.Gamma,self.S)

	def vardrift(self,sig):
		return -np.dot(self.Gamma,sig)-np.dot(sig,self.Gamma.T)+self.H**2

	def getGamma(self):
		g = np.zeros((self.order,self.order))
		for i in range(self.order):
			g[i-1,i] = -1.0
		for i in range(self.order):
			g[-1,i] = binomial(self.order+1,i)*self.gamma**(self.order+1-i)
		return g		

	def getH(self):
		eta = np.zeros((self.order,self.order))
		eta[-1,-1]= self.eta
		return eta	
	
	def geteta(self):
		return self.eta	
	def getgamma(self):
		return self.gamma
	def getstate(self):
		return self.S[:,0]

	def samplestep(self,dt,N =1 ):
		"""Gives a sample of the temporal dependent gp"""
		sample = np.zeros((N,self.order))
		for steps in range(N):
			self.S = self.S + dt*self.drift()+np.sqrt(dt)*np.dot(self.H,np.random.normal(0.0,1.0,(self.order,self.N*self.N)))
			sample[steps,:] = self.S[:,0]
		#sample = np.dot(self.khalf,self.S[-1,:])
		return sample
		#return np.reshape(sample,(self.N,self.N))[::-1]
		
		
class BistableEnv(object):
	"""Implements a gaussian process with correlation structure
	<S_i(t_k)S_j(t_l)> = exp(-(|x(i)-x(j)|/L)**zeta)*phi(|t_k-t_l|;eta,gamma,order)
	we take the points x to be regularly distributed on a grid on the square
	(x0,y0),(x0,y0+Ly),(x0+Lx,y0+Ly),(x0+Lx,y0) and phi is the matern kernel
	with \nu = order-1/2."""
	def __init__(self,zeta,gamma,eta,L,N,x0,y0,Lx,Ly,sigma,order):
		"""Constructer method, generates all the internal data
		xs and ys are the x and y coordinates of all the points.
		zeta, eta, gamma, L are the parameters of the kernel
		N is the number of subdivisions in the grid, sigma is added
		to the diagonal of the gram matrix to ensure positive definiteness
		and order is the order of the matern process
		k is the covariance matrix, and khalf its cholesky decomposition, used to sample"""
		self.xs = np.arange(x0,x0+Lx,Lx/N)
		self.ys = np.arange(y0,y0+Ly,Ly/N)
		self.zeta = zeta
		self.gamma = gamma
		self.eta = eta
		self.L = L
		self.N = N
		self.k = np.zeros((N*N,N*N))
		self.S = np.random.normal(0.0,1.0,(order,N*N))
		self.order = order if order>0 else 1
		if N>1:
			for i in range(0,N*N):
				for j in range(0,N*N):
					(ix,iy) =divmod(i,N)
					(jx,jy) =divmod(j,N)
					dist = np.sqrt((self.xs[ix]-self.xs[jx])**2 + (self.ys[iy]-self.ys[jy])**2) 
					self.k[i,j] = np.exp(-(dist/L)**zeta)+sigma*(i==j)
			self.khalf = np.linalg.cholesky(self.k)
		else:
			self.khalf = np.array([1.0])
	def reset(self,x=None):
		if x:
			self.S = x
		else:
			self.S = np.random.normal(0.0,1.0,(self.order,self.N*self.N))
	
	def drift(self,x=None):
		if x!=None:
			return -2*self.zeta*self.gamma*x**3+2*self.zeta*self.gamma**2*x 
		else:
			return -2*self.zeta*self.gamma*(self.S**3)+2*self.zeta*self.gamma**2*self.S

	def sample(self):
		"""Gets an independent sample from the spatial kernel"""
		s = np.random.normal(0.0,1.0,self.N*self.N)
		s = np.dot(self.khalf,s)
		return np.reshape(s,(self.N,self.N))
	def getgamma(self):
		return self.gamma
	def geteta(self):
		return self.eta	
	
	def getstate(self):
		return self.S[:,0]
	def samplestep(self,dt,N =1 ):
		"""Gives a sample of the temporal dependent gp"""
		sample = np.zeros((N,self.order))
		for steps in range(N):
			self.S[:]= self.S[:]+dt*self.drift()+np.sqrt(dt)*self.eta*np.random.normal(0.0,1.0,(self.order,self.N*self.N))
			sample[steps,:] = self.S[:]
		#sample = np.dot(self.khalf,self.S[-1,:])
		return sample
		#return np.reshape(sample,(self.N,self.N))[::-1]
		
		
