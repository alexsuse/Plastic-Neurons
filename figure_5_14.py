#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python
"""Particle filtering for nonlinear dynamic systems observed through adaptive poisson neurons"""
import particlefilter as pf
import gaussianenv as ge
import poissonneuron as pn
import numpy as np
import cPickle as pic
import multiprocessing as mp
import brewer2mpl
import sys

#parameter definitions

plotting = True


try:
    filename = sys.argv[1]
except:
    filename = 'dense_sparse.pik'


dic = pic.load( open(filename,'r'))
sparse_eps = dic['sparse_eps']
dense_eps = dic['dense_eps']
particle_eps = dic['particle_eps']
alphas = dic['alphas']
dthetas = dic['dthetas']
print "ALL GOOD"

from prettyplotlib import plt
import prettyplotlib as ppl

#matplotlib.rcParams['font.size']=10

xs,ys = np.where(np.isnan(sparse_eps))

for x,y in zip(xs,ys):
    sparse_eps[x,y] = sparse_eps[x-1,y]

max1 = np.max(sparse_eps)
max2 = np.max(dense_eps)
max3 = np.max(particle_eps)

maxtotal = np.max([max1,max2,max3])


min1 = np.min(sparse_eps)
min2 = np.min(dense_eps)
min3 = np.min(particle_eps)

mintotal = np.min([min1,min2,min3])

ax1 = ppl.subplot2grid((3,2),[0,0],rowspan=3)
ax4 = ppl.subplot2grid((3,2),[2,1])
ax2 = ppl.subplot2grid((3,2),[0,1])
ax3 = ppl.subplot2grid((3,2),[1,1])

x = np.arange(-2.0,2.0,0.01)

x_density = np.exp(-x**2/(2*0.25))/np.sqrt(np.pi/2)

alpha=0.5

font = {'size':10}
plt.rc('font',**font)

tuning1 = 0.7*np.exp(-(x-0.8)**2/(2*alpha**2))
tuning2 = 0.7*np.exp(-(x+0.8)**2/(2*alpha**2))

ppl.plot(x,x_density,label="$P(x)$",ax=ax1)
ppl.plot(x,tuning1,label="$\lambda^1(x)$",ax=ax1)
ppl.plot(x,tuning2,label="$\lambda^2(x)$",ax=ax1)

ax1.annotate(r'$\Delta\theta$', xy = (-0.8,0.7), xytext=(0.8,0.692), arrowprops = dict(arrowstyle='<->'))
ax1.annotate(r'$\alpha$', xy = (-0.8-alpha,0.45), xytext=(-0.8+alpha,0.442), arrowprops = dict(arrowstyle='<->'))

ax1.set_xlabel('x')

ppl.legend(ax1,loc=4).get_frame().set_alpha(0.7)

yellorred = brewer2mpl.get_map('YlOrRd','Sequential',9).mpl_colormap

dthetas = 2*dthetas

ddtheta = dthetas[1]-dthetas[0]
dalpha = alphas[1]-alphas[0]

dthetas2,alphas2 = np.meshgrid(np.arange(dthetas.min(),dthetas.max()+0.5*ddtheta,ddtheta)-ddtheta/2,
                               np.arange(alphas.min(),alphas.max()+0.5*dalpha,dalpha)-dalpha/2)

p1 = ax2.pcolormesh(dthetas2,alphas2,dense_eps.T,cmap=yellorred,rasterized=True)
ax2.axis([dthetas2.min(),dthetas2.max(),alphas2.min(),alphas2.max()])
p2 = ax3.pcolormesh(dthetas2,alphas2,sparse_eps.T,cmap=yellorred,rasterized=True)
ax3.axis([dthetas2.min(),dthetas2.max(),alphas2.min(),alphas2.max()])
p3 = ax4.pcolormesh(dthetas2,alphas2,particle_eps.T,cmap=yellorred,rasterized=True)
ax4.axis([dthetas2.min(),dthetas2.max(),alphas2.min(),alphas2.max()])

ticks = np.array([mintotal,(mintotal+maxtotal)/2,maxtotal])
ticklabels = np.round(ticks,decimals=2)

cb1 = plt.colorbar(p1,ax=ax2)
cb1.set_ticks(ticks)
cb1.set_ticklabels(ticklabels)
cb2 = plt.colorbar(p2,ax=ax3)
cb2.set_ticks(ticks)
cb2.set_ticklabels(ticklabels)
cb3 = plt.colorbar(p3,ax=ax4)
cb3.set_ticks(ticks)
cb3.set_ticklabels(ticklabels)

ax4.set_xlabel(r'$\Delta\theta$')
ax4.set_ylabel(r'$\alpha$')
ax3.set_ylabel(r'$\alpha$')
ax2.set_ylabel(r'$\alpha$')

ax3.axes.get_xaxis().set_ticks([])
ax2.axes.get_xaxis().set_ticks([])

#ax3.tick_params(axis='x',which='both',bottom='off')
#ax2.tick_params(axis='x',which='both',bottom='off')

ax2.set_title('ADF with dense coding assumption')
ax3.set_title('Full ADF')
ax4.set_title('Particle Filter')

[p.set_clim(vmin=mintotal,vmax=maxtotal) for p in [p1,p2,p3]]

plt.savefig('figure_5_14.pdf')
