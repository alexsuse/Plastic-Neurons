import matplotlib.pyplot as plt
import numpy as np
import cPickle as pic
plt.rc('text',usetex=True)

fi = open("mmse_adaptive_4","rb")
data = pic.load(fi)

string = [r'$\tau\delta$ = %.2f' % float(0.2*data[2][i]) for i in range(5)]
#string2 = [r'$\tau$ = '+str(data[3][i]) for i in range(5)]
maxmmse = np.max(data[0])
#maxf = np.max(data[1])
def f(x):
	ax1.plot(data[2],data[0][:,x]/maxmmse,label=string[x])
#	ax2.plot(data[2],data[1][:,x]/maxf,label=string2[x])
plt.close()
plt.figure()
ax1 = plt.gcf().add_subplot(1,1,1)
plt.title(r'MMSE for a reconstruction task')
#ax2 = plt.gcf().add_subplot(1,2,2)
#plt.title(r'Firing rate of adaptive code')
map(f,range(5))
ax1.legend()
#ax2.legend()
ax1.set_xlabel(r'$\alpha$')
ax1.set_ylabel(r'MMSE')
#ax2.set_xlabel(r'$\alpha$')
#ax2.set_ylabel(r'Firing Rate')
#plt.xlabel(r'Tuning width $\alpha$')
#plt.ylabel(r'MMSE')
