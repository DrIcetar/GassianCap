from funcs import GassianCap as gc
import copy

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize


""" Ag Absorption  """
# TXDATA = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
# TYDATA = np.array([4.91431, 0.248523, 0.0525253, 0.0185877, 0.00896987,
# 0.00530089, 0.00353163, 0.00251578, 0.00185472, 0.00141267, 0.00108291])
TXDATA = np.array(np.linspace(160, 480, 21))
TYDATA = np.array([1.70597, 1.95145, 2.15909, 2.33655, 2.50163, 2.65016, 2.72534, 2.73848, 2.65879, 2.54181,
                   2.41179, 2.28266, 2.11232, 1.93137, 1.74634, 1.56662, 1.45125, 1.35647, 1.2462, 1.1267, 1])



# XPAR0_FIT = np.array([1, 1, 1])
# initialize the parameter:a mu intervalN ii nn R a b
XPAR0_CAP = np.array([1, 0, 201, 101, 80, 120,1,0])
#bound = ([1, 0, 20, 1, 1, 80], [10, 10, 100, 2, 100, 200])
PAR = optimize.leastsq(gc.error, XPAR0_CAP, args=(TXDATA, TYDATA),maxfev=100000,full_output=1)
# PAR = optimize.least_squares(
#     error, XPAR0_CAP, bounds=bound, args=(TXDATA, TYDATA), max_nfev=1000,
#     verbose=0)
# plt.close()

# plt.plot(TXDATA, gauss_cap(XPAR0_CAP, TXDATA))
# print(error(XPAR0_CAP, TXDATA, TYDATA))
plt.subplot(2, 1, 2)
plt.plot(TXDATA, TYDATA,'r^')

# # # fit plot
print(PAR, 'end of calculation')
# write to txt
#print(np.str(np.around(PAR[0],2)))
output = 'XPars   a   mu  N   ii  nn  R   a   b'\
   + '\nXPar0' + np.str(XPAR0_CAP) + '\nXParF' + np.str(np.around(PAR[0],5)) + '\n\n'
file = open("data.txt",'a')
file.write(output)
#np.savetxt("datasaved.txt",PAR[0])

XFIT = np.linspace(160, 480, 100)
YFIT = gc.gauss_cap(PAR[0], XFIT)
plt.subplot(2,1,2)
plt.plot(XFIT, YFIT[0], 'b-o')
plt.plot(TXDATA,gc.gauss_cap(PAR[0],TXDATA)[0],'r-s')
plt.subplot(2,1,1)
ii = int(np.fix(PAR[0][3]) + 1)
nn = int(np.fix(PAR[0][4]) + 1) + ii
plt.plot(YFIT[2][ii:nn],YFIT[1][ii:nn],'r-s')#gaussian distribution
plt.show()
