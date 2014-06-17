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
    def __init__(self,zeta,gamma,eta,L,N,x0,y0,Lx,Ly,sigma,order,randomstate=None):
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
        if randomstate==None:
            self.rng = np.random
        else:
            self.rng = randomstate
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
            self.S = self.rng.normal(0.0,1.0,(self.order,self.N*self.N))

    def sample(self):
        """Gets an independent sample from the spatial kernel"""
        s = self.rng.normal(0.0,1.0,self.N*self.N)
        s = np.dot(self.khalf,s)    
        return np.reshape(s,(self.N,self.N))

    def drift(self,x=None):
        if x!=None:
            return -np.dot(self.Gamma,np.array([x]))
        else:
            return -np.dot(self.Gamma,self.S)

    def meandrift(self,mu,sig):
        return self.drift(x=mu)

    def vardrift(self,mu,sig):
        return -np.dot(self.Gamma,sig)-np.dot(sig,self.Gamma.T)+self.H**2

    def getGamma(self):
        g = np.zeros((self.order,self.order))
        for i in range(self.order):
            g[i-1,i] = -1.0
        for i in range(self.order):
            g[-1,i] = binomial(self.order,i)*self.gamma**(self.order-i)
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
            self.S = self.S + dt*self.drift()+np.sqrt(dt)*np.dot(self.H,self.rng.normal(0.0,1.0,(self.order,self.N*self.N)))
            sample[steps,:] = self.S[:,0]
        #sample = np.dot(self.khalf,self.S[-1,:])
        return sample
        #return np.reshape(sample,(self.N,self.N))[::-1]


class BistableEnv(object):
    """Generates a bistable environment with a quadric potential given by 
    V(x) = gamma*x**4/4 - gamma*x_0*x**2/2, where x_0 determines the equilibrium points (+- x_0)
    and gamma determines the steepness of the potential.
    """
    def __init__(self,x0,gamma,eta,order,randomstate=None,N=1):
        """Constructer method, generates all the internal data"""
        self.x0 = x0
        self.gamma = gamma
        self.N=N
        self.eta = eta
        self.S = np.random.normal(0.0,1.0,(order,N*N))
        self.order = order if order>0 else 1
        if randomstate==None:
            self.rng = np.random
        else:
            self.rng=randomstate
    def reset(self,x=None):
        if x:
            self.S = x
        else:
            self.S = self.rng.normal(0.0,1.0,(self.order,self.N*self.N))
    
    def drift(self,x=None):
        if x!=None:
            return 4.0*x*(self.x0-x**2).ravel()
        else:
            return 4.0*self.S*(self.x0-self.S**2).ravel()

    def meandrift(self, mu, sig):
        return 4.0*mu*self.x0 - 12*mu*sig - 4.0*mu**3

    def vardrift(self,mu, sig):
        return 8.0*self.x0*sig-24.0*mu**2*sig-24.0*sig**2+self.eta

    def sample(self):
        """Gets an independent sample from the spatial kernel"""
        s = self.rng.normal(0.0,1.0,self.N*self.N)
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
    #        print self.drift()*dt, np.sqrt(dt)*self.eta
            self.S[:]= self.S[:]+dt*self.drift()+np.sqrt(dt)*self.eta*self.rng.normal(0.0,1.0,(self.order,self.N*self.N))
            sample[steps,:] = self.S[:]
        #sample = np.dot(self.khalf,self.S[-1,:])
        return sample
        #return np.reshape(sample,(self.N,self.N))[::-1]
        
        
