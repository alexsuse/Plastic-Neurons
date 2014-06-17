import prettyplotlib as ppl
from prettyplotlib import plt
import numpy as np
import cPickle as pic
import sys
plt.rc('text',usetex=True)

try:
    sys.argv[1]
    data = pic.load(open(sys.argv[1],"rb"))
except:
    fi = open("mmse_bistable_new.pkl","rb")
    data = pic.load(fi)

string = [r'$\phi$ = %.2f' % float(0.2*i) for i in data[2]]
#string2 = [r'$\tau$ = '+str(data[3][i]) for i in range(5)]
maxmmse = np.max(data[0])
#maxf = np.max(data[1])
def f(x):
	ax1.plot(data[2],data[0][:,x],label=string[x])
#	ax2.plot(data[2],data[1][:,x]/maxf,label=string2[x])
fig, ax1 = ppl.subplots(1)
plt.title(r'Estimation')
#ax2 = plt.gcf().add_subplot(1,2,2)
#plt.title(r'Firing rate of adaptive code')
map(f,range(data[2].size))
ax1.legend()
#ax2.legend()
ax1.set_xlabel(r'$\alpha$')
ax1.set_ylabel(r'MMSE')
#ax2.set_xlabel(r'$\alpha$')
#ax2.set_ylabel(r'Firing Rate')
#plt.xlabel(r'Tuning width $\alpha$')
#plt.ylabel(r'MMSE')
plt.savefig("mmse_reconstruction.eps")
