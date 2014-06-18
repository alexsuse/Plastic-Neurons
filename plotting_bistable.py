import prettyplotlib as ppl
from prettyplotlib import plt
import numpy as np
import cPickle as pic
import sys
#plt.rc('text',usetex=True)

try:
    sys.argv[1]
    print "pickling"
    data = pic.load(open(sys.argv[1],"rb"))
except:
    fi = open("mmse_bistable_new.pkl","rb")
    data = pic.load(fi)

print "stringing"
string = [r'$\phi$ = %.2f' % float(i) for i in data[3]]

maxmmse = np.max(data[0])

def f(x):
    line, = ppl.plot(data[2],data[0][:,x],label=string[x],axes=ax1)
    return line.get_color()

def get_min(data):
    minim = np.min(data)
    indmin = np.argmin(data)
    return minim,indmin

print "subplotting"
fig, ax1 = ppl.subplots(1)
ax1.set_title(r'Estimation')

colors = map(f,range(data[3].size))

for i in range(data[3].size):
    mi,ind = get_min(data[0][:,i])
    ppl.plot(data[2][ind],mi,'o',color=colors[i],axes=ax1)

ppl.legend(ax1)

ax1.set_xlabel(r'$\alpha$')
ax1.set_ylabel(r'MMSE')

plt.savefig("mmse_reconstruction.eps")
