# -*- coding: utf-8 -*-
"""
Created on Fri May  2 15:04:22 2014

@author: polina
"""

import numpy as np
from numpy import cos, sin
from simple import Tr
from scipy.optimize import minimize

def get_rho(V, r):
    return 1.0 /(1+r**2) * np.matrix([[0, 0, 0, 0], [0, 1.0, V*r, 0], [0, V*r, r**2, 0], [0, 0, 0, 0]])
    
def get_S_A(eta, rho, alpha, N):
    m = np.matrix([[cos(alpha)**2, 0, cos(alpha)*sin(alpha), 0],\
    [0, cos(alpha)**2, 0, cos(alpha)*sin(alpha)], \
    [cos(alpha)*sin(alpha), 0, sin(alpha)**2, 0], \
    [0, cos(alpha)*sin(alpha), 0, sin(alpha)**2]])
    return N * eta * Tr(m * rho).real + 3000.0
    
def get_S_B(eta, rho, alpha, N):
    m = np.matrix([[cos(alpha)**2, cos(alpha)*sin(alpha), 0, 0],\
    [cos(alpha)*sin(alpha), sin(alpha)**2, 0, 0], \
    [0, 0, cos(alpha)**2, cos(alpha)*sin(alpha)], \
    [0, 0, cos(alpha)*sin(alpha), sin(alpha)**2]])
    return N * eta * Tr(m * rho ).real + 3000.0

def get_C_oo(eta1, eta2, alpha, beta, rho, N):
    m = np.matrix([\
    [cos(alpha)**2*cos(beta)**2, cos(alpha)**2*cos(beta)*sin(beta), cos(alpha)*sin(alpha)*cos(beta)**2, cos(alpha)*sin(alpha)*cos(beta)*sin(beta)],\
    [cos(alpha)**2*cos(beta)*sin(beta), cos(alpha)**2*sin(beta)**2, cos(alpha)*sin(alpha)*cos(beta)*sin(beta), cos(alpha)*sin(alpha)*sin(beta)**2],\
    [cos(alpha)*sin(alpha)*cos(beta)**2, cos(alpha)*sin(alpha)*cos(beta)*sin(beta), sin(alpha)**2*cos(beta)**2, sin(alpha)**2*cos(beta)*sin(beta)],\
    [cos(alpha)*sin(alpha)*cos(beta)*sin(beta), cos(alpha)*sin(alpha)*sin(beta)**2, sin(alpha)**2*cos(beta)*sin(beta), sin(alpha)**2*sin(beta)**2]])
    return N * eta1 * eta2 * Tr(rho * m)
    
def get_n(eta1, eta2, rho):
    N = 24.21 * 10**6
    C_oo_acc = lambda alpha, beta: get_S_A(eta1, rho, alpha, N)*get_S_B(eta2, rho, beta, N) * 6*10**(-10) *\
    (1 - get_C_oo(eta1, eta2, alpha, beta, rho, N)/get_S_A(eta1, rho, alpha, N))*\
    (1 - get_C_oo(eta1, eta2, alpha, beta, rho, N)/get_S_B(eta2, rho, beta, N))
    C_oo = lambda alpha, beta: get_C_oo(eta1, eta2, alpha, beta, rho, N) + C_oo_acc(alpha, beta)
    def n(alpha1, alpha2, beta1, beta2):
        print '-C_oo(alpha1, beta1)={}'.format(-C_oo(alpha1, beta1))
        print 'get_S_A(eta1, rho, alpha1)={}'.format(get_S_A(eta1, rho, alpha1, N))
        print '- C_oo(alpha1, beta2)={}'.format(- C_oo(alpha1, beta2))
        print 'get_S_B(eta2, rho, beta1)={}'.format(get_S_B(eta2, rho, beta1, N))
        print '- C_oo(alpha2, beta1)={}'.format(-C_oo(alpha2, beta1))
        print 'C_oo(alpha2, beta2)={}'.format(C_oo(alpha2, beta2))
        return  (-C_oo(alpha1, beta1) + get_S_A(eta1, rho, alpha1, N) - C_oo(alpha1, beta2)\
    + get_S_B(eta2, rho, beta1, N) - C_oo(alpha2, beta1) + C_oo(alpha2, beta2))
    return n
    
def min_func(eta1, eta2, V, r):
    to_min = lambda x: get_n(eta1, eta2, get_rho(V, r))(x[0], x[1], x[2], x[3])
    res = minimize(to_min, [2*np.pi/3, np.pi/2, 0, np.pi/3], \
    jac=False,  method='Nelder-Mead', options={'disp': True})
    print [res.x[0]/np.pi*180, res.x[1]/np.pi*180, res.x[2]/np.pi*180, res.x[3]/np.pi*180]
    print to_min(res.x)
    print to_min([85.6/180*np.pi, 118.0/180*np.pi, -5.4/180*np.pi, 25.9/180*np.pi])
    
min_func(0.7377, 0.7859, 0.965, 0.297)
