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
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import os


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
 
def get_sigma(theta):
    e = np.e**(-2*1j*theta)
    sigma = np.matrix([[0, e, 0, 0], [np.conjugate(e), 0, 0, 0], [0, 0, 0, e], 
                    [0, 0, np.conjugate(e), 0]])
    return sigma
    
def get_tau(theta):
    e = np.e**(-2*1j*theta)
    tau = np.matrix([[0, 0, e, 0], [0, 0, 0, e], [np.conjugate(e), 0, 0, 0], 
                    [0, np.conjugate(e), 0, 0]])
    return tau
    
def get_n_o(eta, theta, is_first):
    if is_first:
        return eta / 2.0 * (np.eye(4) + get_sigma(theta))
    return eta / 2.0 * (np.eye(4) + get_tau(theta))

def get_n_e(eta, theta, is_first):
    if is_first:
        return eta / 2.0 * (np.eye(4) - get_sigma(theta))
    return eta / 2.0 * (np.eye(4) - get_tau(theta))

def get_n_u(eta, theta, is_first):
    return (1 - eta)   

def get_psi(r, omega):
    return 1 /(2 * np.sqrt(1+r**2)) * np.matrix([[(1+r) * np.e**(-1j*omega)], [-(1-r)], [-(1-r)], [(1+r) * np.e**(1j * omega)]])
    
def get_J(psi, B):
    return Tr(B * psi * psi.transpose().conjugate())
    
def Tr(M):
    if M.shape[0] != M.shape[1]:
        return 0
    sum = 0
    for x in xrange(M.shape[0]):
        sum += M[x, x]
    return sum
    

def get_n(eta1, eta2, zeta):
    n_oo = lambda alpha, beta : get_n_o(eta1, alpha, True) * \
    get_n_o(eta2, beta, False)
    n_oe = lambda alpha, beta : get_n_o(eta1, alpha, True) * \
    get_n_e(eta2, beta, False)
    n_ou = lambda alpha, beta : get_n_o(eta1, alpha, True) * \
    get_n_u(eta2, beta, False)
    n_eo = lambda alpha, beta : get_n_e(eta1, alpha, True) * \
    get_n_o(eta2, beta, False)
    n_uo = lambda alpha, beta : get_n_u(eta1, alpha, True) * \
    get_n_o(eta2, beta, False)
    return lambda theta, d1, d2, d3, d4: n_oe(0 + d1, theta + d4) + n_ou(0 + d1, theta + d4) + n_eo(theta + d2, 0 + d3) + \
    n_uo(theta + d2, 0 + d3) + n_oo(theta + d2, theta + d4) - n_oo(0 + d1, 0 + d3)
    
def J_r_omega(B):
    return lambda x: get_J(get_psi(x[0], x[1]), B(x[2], 0, 0, 0, 0)).real
    
def J_r_omega_delta(B, d1, d2, d3, d4):
    return lambda x: get_J(get_psi(x[0], x[1]), B(x[2], d1, d2, d3, d4))
    
def J_random(x, delta, B):
    if delta == 0:
        return J_r_omega(B)(x)
    r = x[0]
    omega = x[1]
    theta = x[2] % (2*np.pi)
    f = lambda y: J_r_omega(B)([r, omega, theta + y])
    def f_int():
        y = 0
        if (delta != 0):
            y, err = integrate.quad(f, -delta, delta)
        else:
            y = f(0)
        return 1 / 2.0 / delta * y
    return f_int().real
    
def sigma_random(x, delta, B):
    if delta == 0:
        return np.sqrt(get_J(get_psi(x[0], x[1]), B(x[2], 0, 0, 0, 0)**2) - J_random(x, delta, B)**2).real
    f = lambda y: get_J(get_psi(x[0], x[1]), B(x[2] + y, 0, 0, 0, 0)**2)
    def f_int():
        y = 0
        if (delta != 0):
            y, err = integrate.quad(f, -delta, delta)
        else:
            y = f(0)
        return 1 / 2.0 / delta * y
    return np.sqrt(f_int() - J_random(x, delta, B)**2).real

def K(x, delta, B):
    r = J_random(x, delta, B) / sigma_random(x, delta, B)
    #print 'K = {}'.format(r)
    return r
    
J_delta = {}
sigma_delta = {}
K_delta = {}

def key(x, d):
    return (tuple(x.tolist()), d)
    
def J_random_angs(x, delta, B):
    if J_delta.has_key(key(x, delta)):
        print 'j'
        return J_delta[key(x, delta)]
    r = x[0]
    omega = x[1]
    theta = x[2] % (2*np.pi)
    f = lambda y1, y2, y3, y4: J_r_omega_delta(B, y1, y2, y3, y4)([r, omega, theta])
    def f_int():
        y = 0
        if (delta != 0):
            y, err = integrate.nquad(f, [[-delta, delta], [-delta, delta], [-delta, delta], [-delta, delta]])
        else:
            y = f(0)
        return 1 / 16.0 / delta**4 * y
    ret = f_int()
    print 'J = {}'.format(ret)
    J_delta[key(x, delta)] = ret
    return ret
    
def sigma_random_angs(x, delta, B):
    if sigma_delta.has_key(key(x, delta)):
        print 's'
        return sigma_delta[key(x, delta)]
    f = lambda y1, y2, y3, y4: get_J(get_psi(x[0], x[1]), B(x[2], y1, y2, y3, y4)**2)
    def f_int():
        y = 0
        if (delta != 0):
            y, err = integrate.nquad(f,  [[-delta, delta], [-delta, delta], [-delta, delta], [-delta, delta]])
        else:
            y = f(0)
        return 1 / 16.0 / delta**4 * y
    ret = np.sqrt(f_int() - J_random_angs(x, delta, B)**2)
    print 'sigma = {}'.format(ret)
    sigma_delta[key(x, delta)] = ret
    return ret 

def K_angs(x, delta, B):
    if K_delta.has_key(key(x, delta)):
        print 'k'
        return K_delta[key(x, delta)]
    ret = J_random_angs(x, delta, B) / sigma_random_angs(x, delta, B)
    print 'K = {}'.format(ret)
    K_delta[key(x, delta)] = ret
    return ret

def min_func_angs(eta1, eta2, zeta, delta):
    B = get_n(eta1, eta2, zeta)
    #delta = 2.0 / 180 * np.pi

    res1 = minimize(J_r_omega(B), [0, np.pi/3, np.pi/3], \
    jac=False,  method='BFGS', options={'disp': True})
    theta = res1.x[2] % (2*np.pi)
    J_res = np.matrix([res1.x[0], res1.x[1] / np.pi * 180, theta / np.pi * 180, J_r_omega(B)(res1.x), J_random_angs(res1.x, delta, B), sigma_random_angs(res1.x, delta, B), K_angs(res1.x, delta, B)])   

    res2 = minimize(lambda x: J_random_angs(x, delta, B), res1.x, \
    jac=False,  method='Nelder-Mead', options={'disp': True})
    theta = (res2.x[2] / np.pi * 180 ) % 360
    Jdelta_res = np.matrix([res2.x[0], res2.x[1] / np.pi * 180, theta, J_r_omega(B)(res2.x), J_random_angs(res2.x, delta, B), sigma_random_angs(res2.x, delta, B), K_angs(res2.x, delta, B)])
    
    res3 = minimize(lambda x: K_angs(x, delta, B), res1.x, \
    jac=False,  method='Nelder-Mead', options={'disp': True})
    theta = (res3.x[2] / np.pi * 180 ) % 360
    Jsigma_res = np.matrix([res3.x[0], res3.x[1] / np.pi * 180, theta, J_r_omega(B)(res3.x), J_random_angs(res3.x, delta, B), sigma_random_angs(res3.x, delta, B), K_angs(res3.x, delta, B)])

    return [J_res, Jdelta_res, Jsigma_res]
    
def min_func(eta1, eta2, zeta, delta):
    B = get_n(eta1, eta2, zeta)
    #delta = 2.0 / 180 * np.pi

    res1 = minimize(J_r_omega(B), [0, np.pi/3, np.pi/3], \
    jac=False,  method='BFGS', options={'disp': True})
    theta = res1.x[2] % (2*np.pi)
    J_res = np.matrix([res1.x[0], res1.x[1] / np.pi * 180, theta / np.pi * 180, J_r_omega(B)(res1.x), J_random(res1.x, delta, B), sigma_random(res1.x, delta, B), K(res1.x, delta, B)])   

    res2 = minimize(lambda x: J_random(x, delta, B), res1.x, \
    jac=False,  method='Nelder-Mead', options={'disp': True})
    theta = (res2.x[2] / np.pi * 180 ) % 360
    Jdelta_res = np.matrix([res2.x[0], res2.x[1] / np.pi * 180, theta, J_r_omega(B)(res2.x), J_random(res2.x, delta, B), sigma_random(res2.x, delta, B), K(res2.x, delta, B)])
    
    res3 = minimize(lambda x: K(x, delta, B), res1.x, \
    jac=False,  method='Nelder-Mead', options={'disp': True})
    theta = (res3.x[2] / np.pi * 180 ) % 360
    Jsigma_res = np.matrix([res3.x[0], res3.x[1] / np.pi * 180, theta, J_r_omega(B)(res3.x), J_random(res3.x, delta, B), sigma_random(res3.x, delta, B), K(res3.x, delta, B)])

    return [J_res, Jdelta_res, Jsigma_res]
    
    
def simple_min_func(eta1, eta2):
    B = get_n(eta1, eta2, 0)
    #delta = 2.0 / 180 * np.pi

    res1 = minimize(J_r_omega(B), [0, np.pi/3, np.pi/3], \
    jac=False,  method='BFGS', options={'disp': True})
    theta = res1.x[2] % (2*np.pi)
    J_res = np.matrix([eta1, eta2, res1.x[0], res1.x[1] / np.pi * 180, theta / np.pi * 180, J_r_omega(B)(res1.x), -J_r_omega(B)(res1.x)/2])   

    return [J_res]


def print_table(matr):
    to_print = []
    for i in range(matr.shape[0]):
        for j in range(matr.shape[1]):
            separator = ''
            if j != matr.shape[1] - 1:
                separator = ' & '
            new_line = ''
            #if (i != matr.shape[0] - 1) and (j == matr.shape[1] - 1):
            if j == matr.shape[1] - 1:
                new_line = '\\\\\\hline\n'
            to_print += [matr[i,j], separator, new_line]
    str = '{:.4g}{}{}' * matr.shape[0]*matr.shape[1]
    print str.format(*to_print)

#prints every matrix from list as a separate row with common columns
def print_table_row(matr_list, common):
    c = len(common)
    n = matr_list[0].shape[1] + c
    to_print = []
    str = '{}' * c + '{}{:.5g}{}' * matr_list[0].shape[1] * len(matr_list)
    j = 0
    for col in common:
        to_print.append('\\multirow{{{}}}{{*}}{{{}}} & '.format(len(matr_list), col))
    for m in matr_list:
        for i in range(m.shape[1]):
            separator = ''
            if i != 0 or j != 0:
                separator = ' & '
            if i == 0 and j != 0:
                separator = ' & ' * c
            new_line = ''
            if i == m.shape[1] - 1:
                if j != len(matr_list) - 1:
                    new_line = '\\\\\\cline{{{}-{}}}\n'.format(c+1, n)
                else:
                    new_line = '\\\\\\hline\n'
            to_print += [separator, m[0, i], new_line]
        j += 1
    return (to_print, str)

def print_complex_table(rows_list):
    to_print = []
    str = '\\\\\\hline\n'
    for p,s in rows_list:
        to_print += p
        str += s
    print str.format(*to_print)
    
    
def show_plot(x, y, xlabel, ylabel, title, filename):
    directory = './imgs/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.rc('text', usetex=True)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
#    plt.title(title)
    plt.grid()
    plt.savefig(directory + filename + '.eps', format='eps')
    plt.show()
    
def show_multiple_plot(x, ys, xlabel, ylabel, legend, filename):
    directory = './imgs/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.rc('text', usetex=True)
    for y in ys:
        plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend(legend, loc=8)
    plt.savefig(directory + filename + '.eps', format='eps')
    plt.show()
    
def show_plot3d(x, y, z, xlabel, ylabel, zlabel, title, filename):
    directory = './imgs/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.rc('text', usetex=True)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    fig.colorbar(surf, shrink=0.7, aspect=5)
    #plt.title(title)
    plt.grid()
    plt.savefig(directory + filename + '.eps', format='eps')
    plt.show()
        

def eberhard_repeat():
    i = 0
    to_print = []
    r = []
    omega = []
    theta = []
    J = []
    etas = [0.01 * e for e in range(70, 101, 5)]
    for e1 in etas:
        res = simple_min_func(e1, e1)[0]
        to_print.append(print_table_row([res], []))
        r.append(res[0,2])
        omega.append(res[0,3])
        theta.append(res[0,4])
        J.append(res[0,5])
        i += 1
    print_complex_table(to_print)
    
    #show_plot(etas, r, r'$\eta$', r'$r$', r'optimal $r$ for different efficiencies', 'r')
    #show_plot(etas, omega, r'$\eta$', r'$\omega$', r'optimal $\omega$ for different efficiencies', 'omega')
    #show_plot(etas, theta, r'$\eta$', r'$\theta$', r'optimal $\theta$ for different efficiencies', 'theta')
    #show_plot(etas, J, r'$\eta$', r'$J$', r'optimal $J$ for different efficiencies', 'J')

def eberhard_different_etas():
    i = 0
    to_print = []
    count = (36, 36)
    r = np.zeros(count)
    omega = np.zeros(count)
    theta = np.zeros(count)
    J = np.zeros(count)
    etas = [0.01 * e for e in range(65, 101, 5)]
    for e1 in etas:
        j = 0
        for e2 in etas:
            res = simple_min_func(e1, e2)[0]
            to_print.append(print_table_row([res], []))
            r[i, j] = res[0,2]
            omega[i, j] = res[0,3]
            theta[i, j] = res[0,4]
            J[i, j] = res[0,5]
            j += 1
        i += 1
    print_complex_table(to_print)
    [X, Y] = np.meshgrid(etas, etas)
    
    #show_plot3d(X, Y, r, r'$\eta_1$', r'$\eta_2$', r'$r$', r'optimal $r$ for different efficiencies', 'r3d')
    #show_plot3d(X, Y, omega, r'$\eta_1$', r'$\eta_2$', r'$\omega$', r'optimal $\omega$ for different efficiencies', 'omega3d')
    #show_plot3d(X, Y, theta, r'$\eta_1$', r'$\eta_2$', r'$\theta$', r'optimal $\theta$ for different efficiencies', 'theta3d')
    #show_plot3d(X, Y, J, r'$\eta_1$', r'$\eta_2$', r'$J$', r'optimal $J$ for different efficiencies', 'J3d')
    
def one_delta():
    i = 0
    to_print = []
    count = (5, 7)
    r = np.zeros(count)
    omega = np.zeros(count)
    theta = np.zeros(count)
    J = np.zeros(count)
    K = np.zeros(count)
    etas = [0.01*x for x in range(70, 101, 5)]
    for e1 in etas:
        res = simple_min_func(e1, e1)[0]
        r[0, i] = res[0,2]
        omega[0, i] = res[0,3]
        theta[0, i] = res[0,4]
        J[0, i] = res[0,5]
        i = i + 1
    i = 0
    for e1 in etas:
        for d in range(1, 5):
            res = min_func(e1, e1, 0, 0.25 * d * np.pi / 180)
            to_print.append(print_table_row(res, [e1, 0.25 * d ]))
            j = d
            r[j, i] = res[2][0, 0]
            omega[j, i] = res[2][0, 1]
            theta[j, i] = res[2][0, 2]
            J[j, i] = res[2][0, 4]
            K[j, i] = res[2][0, 6]
        i += 1
    print_complex_table(to_print)
    show_multiple_plot(etas, r, r'$\eta$', r'$r$', (r'$\delta = 0$', r'$\delta = 0.25$',r'$\delta = 0.5$', r'$\delta = 0.75$', r'$\delta = 1$' ), 'r_ang')
    show_multiple_plot(etas, omega, r'$\eta$', r'$\omega$', (r'$\delta = 0$', r'$\delta = 0.25$',r'$\delta = 0.5$', r'$\delta = 0.75$', r'$\delta = 1$' ), 'omega_ang')
    show_multiple_plot(etas, theta, r'$\eta$', r'$\theta$', (r'$\delta = 0$', r'$\delta = 0.25$',r'$\delta = 0.5$', r'$\delta = 0.75$', r'$\delta = 1$' ), 'theta_ang')
    show_multiple_plot(etas, J, r'$\eta$', r'$J$', (r'$\delta = 0$', r'$\delta = 0.25$',r'$\delta = 0.5$', r'$\delta = 0.75$', r'$\delta = 1$' ), 'J_ang')

def multilple_deltas():
    i = 0
    to_print = []
    count = (2, 7)
    r = np.zeros(count)
    omega = np.zeros(count)
    theta = np.zeros(count)
    J = np.zeros(count)
    K = np.zeros(count)
    etas = [0.01*x for x in range(70, 101, 5)]
    for e1 in etas:
        res = simple_min_func(e1, e1)[0]
        r[0, i] = res[0,2]
        omega[0, i] = res[0,3]
        theta[0, i] = res[0,4]
        J[0, i] = res[0,5]
        i = i + 1
    i = 0
    r[1, :] = np.array([0.137124, 0.313658, 0.469841, 0.61123, 0.743038, 0.871004, 0.999999])
    omega[1,:] = np.array([3.42997, 9.91344, 15.2419, 18.9498, 21.167, 22.2272, 22.4981])
    theta[1, :] = np.array([21.496, 32.2158, 38.3231, 41.9292, 43.8961, 44.7823, 44.998])
    J[1, :] = np.array([-0.000453514, -0.00614786, -0.0218953, -0.0496699, -0.0898969, -0.142435, -0.207107])
    show_multiple_plot(etas, r, r'$\eta$', r'$r$', (r'$\delta = 0$', r'$\delta = 0.25$',r'$\delta = 0.5$', r'$\delta = 0.75$', r'$\delta = 1$' ), 'r_4ang')
    show_multiple_plot(etas, omega, r'$\eta$', r'$\omega$', (r'$\delta = 0$', r'$\delta = 0.25$',r'$\delta = 0.5$', r'$\delta = 0.75$', r'$\delta = 1$' ), 'omega_4ang')
    show_multiple_plot(etas, theta, r'$\eta$', r'$\theta$', (r'$\delta = 0$', r'$\delta = 0.25$',r'$\delta = 0.5$', r'$\delta = 0.75$', r'$\delta = 1$' ), 'theta_4ang')
    show_multiple_plot(etas, J, r'$\eta$', r'$J$', (r'$\delta = 0$', r'$\delta = 0.25$',r'$\delta = 0.5$', r'$\delta = 0.75$', r'$\delta = 1$' ), 'J_4ang')


#eberhard_repeat()
#one_delta()
#eberhard_different_etas()
multilple_deltas()


