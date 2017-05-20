"""   GaussCap   """

import copy

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize

#import matplotlib.animation as animation

def Sphere(r, nmesh, center):
    """   Create a Sphere at given center   """
    theta = np.array([np.linspace(0, np.pi, nmesh + 1)]).T
    phi = np.array([np.linspace(0, 2 * np.pi, nmesh + 1)])
    x = center[0] + r * np.sin(theta) * np.sin(phi)
    y = center[1] + r * np.sin(theta) * np.cos(phi)
    z = center[2] + r * np.cos(theta) * np.array([np.ones(nmesh + 1)])
    return (x, y, z)


def gauss_fit(xpar, xdata):
    """    Gauss Fit    """
    print("a	mu	interval	intervalN	ii	nn	R	sphere	exp")
    # print(xpar.real)
    plt.clf()
    y = xpar[0] * np.e ** (xpar[1] / xdata) + xpar[2]
    plt.plot(xdata, y, 'r-', TXDATA, TYDATA, 'bo')
    plt.pause(0.0001)
    return y


def gauss_cap(xpar, xdata):
    """   GaussCap   """
    print(['a mu intervalN ii nn R sphere a b'],'\n',xpar)

    ########################################### Gaussian Distribution ########
    #plt.clf()
    sigma = xpar[0]
    mu = xpar[1]
    a = xpar[6]
    b = xpar[7]
    t = np.linspace(mu - 3 * sigma, mu + 3 * sigma, np.fix(xpar[2]) + 2)
    g = b + a * np.exp(-(t - mu) ** 2 / (2 * sigma * sigma)) / \
        (np.sqrt(2 * np.pi) * sigma)
    #plt.subplot(211)
    #plt.plot(t, g, 'r-s')
    #plt.xlabel('t')
    #plt.ylabel('g')
    #plt.show()
    ########################################### Sphere #######################
    ii = int(np.fix(xpar[3]) + 1)
    nn = int(np.fix(xpar[4]) + 1)
    R = np.linspace(1, xpar[5], nn)
    sphere = Sphere(1, 100, (0, 0, 0))
    xg = []
    yg = []
    zg = []
    sg = g[ii:ii + nn - 1]  # maker size, ii+nn not included
    cg = np.arange(nn, 1, -1)  # maker color , 0 not included
    for i in range(1, nn):
        xg = np.append(xg, R[i] * sphere[0])
        yg = np.append(yg, R[i] * sphere[1])
        zg = np.append(zg, R[i] * sphere[2])
    sgr = np.reshape(np.tile(sg, (np.size(sphere[0]), 1)).T, xg.size)
    cgr = np.reshape(np.tile(cg, (np.size(sphere[0]), 1)).T, xg.size)
    # print(xg.size,sg.size,cg.size,np.size(sphere[0]),sgr.size)
    # print(xg.shape,sgr.shape,type(sgr),'\n',xg[1:150])
    # print(np.max(yg))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # # # ax.scatter(sphere[0], sphere[1], sphere[2])
    # ax.scatter(xg, yg, zg, sgr, cgr)
    # plt.show()
    ########################################### QW-eBeam #####################
    qw_period = 20
    qw_width = 3
    ebeam_line = 680 - xdata
    V = np.ones(len(xdata))
    for i in range(1, len(ebeam_line)):
        xx = copy.deepcopy(xg)  # check copy and deepcopy
        yy = copy.deepcopy(yg)
        zz = copy.deepcopy(zg)
        ss = copy.deepcopy(sgr)
        cc = copy.deepcopy(cgr)
        #print(ss[0:10])
        # print(id(cc), id(cgr), cc is cgr)
        # print(ebeam_line[i])
        ebeam_sphere = yg + ebeam_line[i]
        # print(len(ebeam_sphere),len(xx),len(yy),len(zz),len(ss),len(cc))
        # print(i, ebeam_sphere.shape, ebeam_line[i], np.sum(
        #     ebeam_sphere), yy[15:20], cc[10:15])
        #print("i:",i)
        for pos in range(len(ebeam_sphere)):
            if ebeam_sphere[pos] < 300 or ebeam_sphere[pos] > 502:
                # print(ebeam_sphere[pos])
                xx[pos] = 0
                yy[pos] = 0
                zz[pos] = 0
                ss[pos] = 0
                cc[pos] = 0
            elif ebeam_sphere[pos] % qw_period > qw_width or ebeam_sphere[pos] % qw_period == 0:
                # print(ebeam_sphere[pos])
                xx[pos] = 0
                yy[pos] = 0
                zz[pos] = 0
                ss[pos] = 0
                cc[pos] = 0
            else:
                # print(ebeam_sphere[pos])
                #yy[pos] = ebeam_sphere[pos]
                #print(yy[pos],ss[pos])
                V[i] = V[i] + ss[pos]
                # print(V[i])
        #print(V[i])
        # plt.plot(yy) fig = plt.figure() ax = fig.add_subplot(111,
        # projection='3d') ax.scatter(xx, yy, zz, ss, cc)
        # ax.view_init(elev=90,azim=0) plt.show() V[i] = np.sum(ss)
    #print("V:",V)
    return (V,g,t)
    ########################################### QW-eBeam #####################
def error(xpar, xdata, ydata):
    """ error calculation   """
    return (gauss_cap(xpar, xdata)[0] - ydata)


