#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python
"""Particle filtering for nonlinear dynamic systems observed through adaptive poisson neurons"""
import particlefilter as pf
import gaussianenv as ge
import poissonneuron as pn
import numpy as np
import cPickle as pic
import multiprocessing as mp

#parameter definitions

plotting = True

dthetas = np.arange(0.1,2.0,0.1)
alphas = np.arange(0.1,2.0,0.1)

sparse_eps = np.zeros((dthetas.size,alphas.size))
dense_eps = np.zeros((dthetas.size,alphas.size))
particle_eps = np.zeros((dthetas.size,alphas.size))

def run_filters(args):

    i,j,dtheta,alpha = args
    dt = 0.001
    phi = 1.0
    zeta = 1.0
    eta = 1.0
    gamma = 1.0
    timewindow = 20000
    dm = 0.0
    tau = 1.0
    nparticles = 500
    plotting = True

    #env is the "environment", that is, the true process to which we don't have access

    env_rng = np.random.mtrand.RandomState()

    env = ge.GaussianEnv(gamma=gamma,eta=eta,zeta=zeta,x0=0.0,
                         y0=.0,L=1.0,N=1,order=1,sigma=0.1,
                         Lx=1.0,Ly=1.0,randomstate=env_rng)
    env.reset(np.array([0.0]))

    #code is the population of neurons, plastic poisson neurons

    code_rng = np.random.mtrand.RandomState()

    code = pn.PoissonPlasticCode(phi=phi,tau=tau,alpha=alpha
                                 thetas=np.arange(-dtheta,dtheta+0.1,2*dtheta),
                                 dm=dm,randomstate=code_rng)
    
    env_rng.seed(12345)
    code_rng.seed(67890)
    env.reset(np.array([0.0]))
    code.reset()
    
    [densem,densevar,spsg,sg,dense_mse] = pf.gaussian_filter(code,env,timewindow=timewindow,dt=dt,
                                                            dense=True)
    
    env_rng.seed(12345)
    code_rng.seed(67890)
    env.reset(np.array([0.0]))
    code.reset()
    
    [sparsem,sparsevar,spsg,sg,sparse_mse] = pf.gaussian_filter(code,env,timewindow=timewindow,dt=dt,
                                                                dense=False)
    
    env_rng.seed(12345)
    code_rng.seed(67890)
    env.reset(np.array([0.0]))
    code.reset()
    
    [msep,ws] = pf.mse_particle_filter(code, env, timewindow=timewindow,
                                                         dt=dt, nparticles=nparticles,
                                                         testf=(lambda x:x))
    return i,j,dense_mse,sparse_mse,particle_mse

try:
    dic = pic.load( open('dense_sparse.pik','r'))
    sparse_eps = dic['sparse_eps']
    dense_eps = dic['dense_eps']
    particle_eps = dic['particle_eps']
    alphas = dic['alphas']
    dthetas = dic['dthetas']
    print "ALL GOOD"

except:

    args = []

    for i,dtheta in enumerate(dthetas):
        for j,alpha in enumerate(alphas):
            args.append((i,j,dtheta,alpha))

    p = mp.Pool(mp.cpu_count())

    results = p.map(run_filters,args)

    for res in results:
        i,j,dense_mse,sparse_mse,msep = res

        dense_eps[i,j] = dense_mse
        sparse_eps[i,j] = sparse_mse
        particle_eps[i,j] = msep

    with open("dense_sparse.pik","wb") as f:
        pic.dump({'dense_eps':dense_eps,
                  'sparse_eps':sparse_eps,
                  'particle_eps':particle_eps,
                  'dthetas':dthetas,
                  'alphas':alphas},
                  f)

if plotting:
    from prettyplotlib import plt
    import prettyplotlib as ppl
    
    #matplotlib.rcParams['font.size']=10
   
    fig, (ax1,ax2,ax3) = ppl.subplots(1,3)

    p1 = ax1.pcolormesh(dense_eps)
    plt.colorbar(p1,ax=ax1)
    p2 = ax2.pcolormesh(sparse_eps)
    p3 = ax3.pcolormesh(particle_eps)


    plt.show()
