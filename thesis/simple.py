# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 22:04:24 2014

@author: polina
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


#def get_B(eta, zeta, alpha1, alpha2, beta1, beta2):
#    A = eta / 2.0 * (np.e**(2.0 * 1j * (alpha1-alpha2)) - 1)
#    B =  np.e**(2.0 * 1j * (beta1-beta2)) - 1
#    tmp = 1 - eta
#    ksi = 4 * zeta / eta
#    M = eta / 2.0 * np.matrix([[2 - eta + ksi, tmp, tmp, A.conjugate()*B.conjugate() - eta],\
#                               [tmp, 2 - eta + ksi, A*B.conjugate() - eta, tmp], \
#                               [tmp, A.conjugate()*B - eta, 2 - eta + ksi, tmp],\
#                               [A*B - eta, tmp, tmp, 2 - eta + ksi]])
#    return M

def get_B(eta, zeta, theta):
    A = eta / 2.0 * (np.e**(2.0 * 1j * theta) - 1)
    B =  np.e**(2.0 * 1j * theta) - 1
    tmp = 1 - eta
    ksi = 4 * zeta / eta
    M = eta / 2.0 * np.matrix([[2 - eta + ksi, tmp, tmp, A.conjugate()*B.conjugate() - eta],\
                               [tmp, 2 - eta + ksi, A*B.conjugate() - eta, tmp], \
                               [tmp, A.conjugate()*B - eta, 2 - eta + ksi, tmp],\
                               [A*B - eta, tmp, tmp, 2 - eta + ksi]])
    return M
    

def get_psi(r, omega):
    return 1 /(2 * np.sqrt(1+r**2)) * np.matrix([[(1+r) * np.e**(-1j*omega)], [-(1-r)], [-(1-r)], [(1+r) * np.e**(1j * omega)]])
    
def get_J(psi, B):
    return Tr(B * psi * psi.transpose().conjugate())#(psi.transpose().conjugate() * B * psi)[0, 0]
    
def Tr(M):
    if M.shape[0] != M.shape[1]:
        return 0
    sum = 0
    for x in xrange(M.shape[0]):
        sum += M[x, x]
    return sum
    
    
alpha1 = 85.6/180*np.pi
alpha2 = 118.0/180*np.pi
beta1 = -5.4/180*np.pi
beta2 = 25.9/180*np.pi
r = 0.297
eta = 0.7
zeta = 0.0
V = 0.965

#omega = -9.7/180*np.pi
B = lambda theta: get_B(eta, zeta, theta)
#psi = get_psi(r, omega)
#print get_J(psi, B)
# J(omega) with Zeilinger parameters
J_omega = lambda o, t: get_J(get_psi(r, o[0]), B(t))
#x = [1.0/180 * np.pi * xx for xx in range(0, 360, 2)]
#y = [J_omega([o]) for o in x]
#plt.plot(x, y)
#plt.grid()
#res = minimize(J_omega, [np.pi/3], \
#jac=False,  method='Nelder-Mead')
#print 'min_x = {}, J = {}'.format(res.x / np.pi * 180, J_omega(res.x))

# using rho from Zeilinger
#psi = 1 /np.sqrt(1+r**2) * np.matrix([[0], [1], [r], [0]])
#rho = psi * psi.transpose()
#for x in xrange(rho.shape[0]):
#    for y in xrange(rho.shape[1]):
#        if x != y:
#            rho[x, y] *= V
#J2 = Tr(rho * B)
#print J2

#optimization J(r, omega)
J_r_omega = lambda x: get_J(get_psi(x[0], x[1]), B(x[2]))
res = minimize(J_r_omega, [0.311, -9.76 / 180.0 * np.pi, 32], \
jac=False,  method='BFGS', options={'disp': True})
print 'min_x = {}, J = {}'.format([res.x[0], res.x[1] / np.pi * 180, res.x[2] / np.pi * 180], J_r_omega(res.x))

def J_random(x):
    r = x[0]
    omega = x[1]
    theta = x[2]
    f = lambda y: J_r_omega([r, omega, theta + y])
    def f_int(delta):
        delta = 0.05
        y = 0
        if (delta != 0):
            y, err = integrate.quad(f, -delta, delta)
        else:
            y = f(0)
        print 1 / 2.0 / delta * y
        return 1 / 2.0 / delta * y
    return f_int(x[3])
    
def sigma_random(x):
    r = x[0]
    f = lambda y: get_J(get_psi(x[0], x[1]), B(x[2] + y)**2)
    def f_int(delta):
        delta = 0.05
        y = 0
        if (delta != 0):
            y, err = integrate.quad(f, -delta, delta)
        else:
            y = f(0)
        print 1 / 2.0 / delta * y
        return 1 / 2.0 / delta * y
    return f_int(x[3]) - J_random(x)**2


def f(x):
    print 'sigma = {}'.format(sigma_random(x))
    return J_random(x)

res = minimize(f, [0.311, -9.76 / 180.0 * np.pi, 32, 1], \
jac=False,  method='BFGS', options={'disp': True})
print 'min_x = {}, J = {}'.format([res.x[0], res.x[1] / np.pi * 180, res.x[2] / np.pi * 180, res.x[3]], J_random(res.x))



sigma = lambda x: np.sqrt(get_J(get_psi(x[0], x[1]), B(x[2])**2) - J_r_omega(x)*J_r_omega(x))
K = lambda x: J_r_omega(x) / sigma(x)

#x = [0.98 * res.x[0] + 0.00005 * res.x[0] * xx for xx in range(0, 800)]
#y = [(0.98 * res.x[1] + 0.00005 * res.x[1] * xx) / np.pi * 180 for xx in range(0, 800)]
#X, Y = np.meshgrid(x, y)
#z = np.zeros(X.shape)
#for x in xrange(X.shape[0]):
#    for y in xrange(X.shape[1]):
#        z[x, y] = sigma([x, y/ 180.0 * np.pi])
#print z.min()
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot_surface(X, Y, z)
#plt.grid()
#plt.show()

#res = minimize(K, [res.x[0], res.x[1], res.x[2]], \
#jac=False,  method='BFGS', options={'disp': True})
#print 'min_x = {}, J = {}'.format([res.x[0], res.x[1] / np.pi * 180,  res.x[2] / np.pi * 180], K(res.x))
#
#print K( [res.x[0], res.x[1], res.x[2]])
#print sigma( [res.x[0], res.x[1], res.x[2]])