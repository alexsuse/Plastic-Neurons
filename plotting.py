##import matplotlib
##matplotlib.use('Agg')
##import matplotlib.pyplot as plt
import prettyplotlib as ppl
from prettyplotlib import plt
import numpy as np
import cPickle as pic
import sys
plt.rc('text',usetex=True)

try:
    sys.argv[1]
    data = pic.load(open(sys.argv[1],"rb"))
    #data = np.load(open(sys.argv[1]))
except:
    fi = open("../data/pickle_alpha_1","rb")
    data = np.load(fi)

taus = data[3]
alphas = data[2]
eps = data[0]
delta = 0.1

string = [r'$\tau\delta$ = %.2f' % float(delta*t) for t in taus]
#string2 = [r'$\tau$ = '+str(data[3][i]) for i in range(5)]
maxmmse = np.max(eps)
#maxf = np.max(data[1])
def f(x):
    ppl.plot(alphas,eps[:,x]/maxmmse,label=string[x],ax=ax1)
#	ax2.plot(data[2],data[1][:,x]/maxf,label=string2[x])
#plt.close()
#plt.figure()
#ax1 = plt.gcf().add_subplot(1,1,1)
fig, ax1 = ppl.subplots(1)
plt.title(r'MMSE for a Reconstruction Task with Adaptive Neurons')
#ax2 = plt.gcf().add_subplot(1,2,2)
#plt.title(r'Firing rate of adaptive code')
map(f,range(0,10,2))
ppl.legend(ax1)
#ax2.legend()
ax1.set_xlabel(r'$\alpha$')
ax1.set_ylabel(r'MMSE')
#ax2.set_xlabel(r'$\alpha$')
#ax2.set_ylabel(r'Firing Rate')
#plt.xlabel(r'Tuning width $\alpha$')
#plt.ylabel(r'MMSE')
plt.show()
#plt.savefig("mmse_discrimination.png",dpi=600)
plt.savefig("figure_5_8.png",dpi=600)
plt.savefig("figure_5_8.eps")
