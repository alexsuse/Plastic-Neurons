import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def f(eps,gamma,eta,phi,alpha,thetas,tau,delta):
    summ = g(eps,thetas,gamma,eta,phi,alpha,tau,delta)
    return -2.0*gamma*eps+eta**2-phi*summ*eps**2/(np.sqrt(1.0+eps/alpha**2)*(alpha**2+eps)**2)


def k(thetas,gamma,eta,phi,alpha,tau,delta):
    numer = 1.0+tau*phi*delta*numpy.exp(-thetas**2/(2*alpha**2+eta**2/gamma))/np.sqrt(1.0+eta**2/(2*gamma))
    return 1.0/numer

def g(eps,thetas,gamma,eta,phi,alpha,tau,delta):    
    Cm = np.exp(-thetas**2/(2*alpha**2+2*eps))
    numer = 1.0+tau*phi*delta*np.exp(-thetas**2/(2*alpha**2+eta**2/gamma))/np.sqrt(1.0+eta**2/(2*gamma*alpha**2))
    summ = np.sum(Cm*thetas**2/numer)        
    return summ    

def lambdabar(thetas,gamma,eta,phi,alpha,tau,delta):    
    Cm = np.exp(-thetas**2/(2*alpha**2+eta**2/gamma))
    numer = 1.0+tau*phi*delta*np.exp(-thetas**2/(2*alpha**2+eta**2/gamma))/np.sqrt(1.0+eta**2/(2*gamma))
    summ = np.sum(Cm/numer)/np.sqrt(1.0+eta**2/(2*gamma*alpha**2))        
    return phi*summ    

def get_mf_gaussian_eps(gamma,eta,phi,alpha,thetas,tau,delta):
    fun = lambda x : f(x,gamma,eta,phi,alpha,thetas,tau,delta)
    x0 = eta**2/(2*gamma)

gamma = 1.0
eta = 1.0
alpha = 1.0
delta = 0.1
tau = 1.0
#phi = 1.0
thetas = np.arange(-4.0,4.0,0.2)
deltas = np.arange(0.0,1.0,0.2)
phis = np.arange(0.0,100.0,0.1)
alpha = 0.5
#alphas = np.arange(0.001,4.0,0.05)
ndeltas = len(deltas)
nphis = len(phis)
#nalphas = len(alphas)
epss = np.zeros((ndeltas,nphis))
ls = np.zeros((ndeltas,nphis))

if __name__=='__main__':
    
    plt.rc('text',usetex=True)
    try:
        data = np.load('../data/mean_field_eps.npz')
        eps = data['epss']
        alphas = data['alphas']
        deltas = data['deltas']
        tau = data['tau']
    
    except:
        for nd,delta in enumerate(deltas):
            y = eta**2/(2*gamma)
            print nd, " of ",len(deltas)
            for nphi,phi in enumerate(phis):
                fun = lambda x: f(x,gamma,eta,phi,alpha,thetas,tau,delta)
                y = opt.fsolve(fun, y)
                epss[nd,nphi] = y
                ls[nd,nphi] = lambdabar(thetas,gamma,eta,phi,alpha,tau,delta)
    
    strings = [r'$\tau\delta$ = '+str(i*tau) for i in deltas]
    
    #f = lambda i : plt.plot(alphas,epss[i,:],label=strings[i])
    #map(f,range(len(deltas)))
    fig, ax = ppl.subplots(1)
    ppl.plot(ls[0,:],epss[0,:],label = strings[0], ax=ax)
    ppl.plot(ls[1,:],epss[1,:],label = strings[1], ax=ax)
    ppl.plot(ls[2,:],epss[2,:],label = strings[2], ax=ax)
    ppl.plot(ls[3,:],epss[3,:],label = strings[3], ax=ax)
    ppl.plot(ls[4,:],epss[4,:],label = strings[4], ax=ax)
    
    np.savez(file = "../data/mean_field_ratedist", eps = epss, phis = phis, deltas = deltas, tau = tau)
    
    ax.xlabel(r'$\lambda$ in spikes/sec')
    ax.ylabel(r'MMSE/$\epsilon_0$')
    ax.title("Rate-Distortion Curve for Adaptive Neurons")
    ax.legend(ax)
    plt.show()
    plt.savefig("rate_distortion_mf_alpha.png")
