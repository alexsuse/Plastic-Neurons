import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def f(eps,gamma,eta,phi,alpha,thetas,tau,delta):
	summ = g(thetas,gamma,eta,phi,alpha,tau,delta)
	return -2.0*gamma*eps+eta**2-phi*summ*eps**2/(np.sqrt(1.0+eps/alpha**2)*(alpha**2+eps)**2)


def k(thetas,gamma,eta,phi,alpha,tau,delta):
	numer = 1.0+tau*phi*delta*numpy.exp(-thetas**2/(2*alpha**2+eta**2/gamma))/np.sqrt(1.0+eta**2/(2*gamma))
	return 1.0/numer

def g(thetas,gamma,eta,phi,alpha,tau,delta):	
	Cm = np.exp(-thetas**2/(2*alpha**2+eta**2/gamma))
	numer = 1.0+tau*phi*delta*np.exp(-thetas**2/(2*alpha**2+eta**2/gamma))/np.sqrt(1.0+eta**2/(2*gamma))
	summ = np.sum(Cm*thetas**2/numer)		
	return summ	

def lambdabar(thetas,gamma,eta,phi,alpha,tau,delta):	
	Cm = np.exp(-thetas**2/(2*alpha**2+eta**2/gamma))
	numer = 1.0+tau*phi*delta*np.exp(-thetas**2/(2*alpha**2+eta**2/gamma))/np.sqrt(1.0+eta**2/(2*gamma))
	summ = np.sum(Cm/numer)/np.sqrt(1.0+eta**2/(2*gamma*alpha**2))		
	return phi*summ	

gamma = 1.0
eta = 1.0
alpha = 1.0
delta = 0.1
tau = 1.0
thetas = np.arange(-4.0,4.0,0.2)
deltas = np.arange(0.0,0.6,0.1)
phis = np.arange(0.0,1000.0,0.1)
ndeltas = len(deltas)
nphis = len(phis)
epss = np.zeros((ndeltas,nphis))
ls = np.zeros((ndeltas,nphis))

plt.rc('text',usetex=True)

for nd,delta in enumerate(deltas):
	for nphi,phi in enumerate(phis):
		print nd, " of ",len(deltas)
		fun = lambda x: f(x,gamma,eta,phi,alpha,thetas,tau,delta)
		y = opt.fsolve(fun, eta**2/(2*gamma))
		epss[nd,nphi] = y
		ls[nd,nphi] = lambdabar(thetas,gamma,eta,phi,alpha,tau,delta)

def plot():
strings = [r'$\tau\delta$ = '+str(i*tau) for i in deltas]
	
plt.plot(ls[0,:],epss[0,:],label = strings[0])
plt.plot(ls[1,:],epss[1,:],label = strings[1])
plt.plot(ls[2,:],epss[2,:],label = strings[2])
plt.plot(ls[3,:],epss[3,:],label = strings[3])
plt.plot(ls[4,:],epss[4,:],label = strings[4])
plt.xlabel(r'$\lambda$ in spikes/sec')
plt.ylabel(r'MMSE/$\epsilon_0$')
plt.title("Rate-Distortion Curve for Adaptive Neurons")
plt.legend()
