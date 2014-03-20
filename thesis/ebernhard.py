# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:15:12 2013

@author: polina
"""

from sympy import * 
import numpy as np
from scipy.optimize import minimize
import pylab as pl

from function_tricks import *

def get_sigma(theta):
    e = E**(-2*I*theta)
    sigma = Matrix([[0, e, 0, 0], [conjugate(e), 0, 0, 0], [0, 0, 0, e], 
                    [0, 0, conjugate(e), 0]])
    return sigma
    
def get_tau(theta):
    e = E**(-2*I*theta)
    tau = Matrix([[0, 0, e, 0], [0, 0, 0, e], [conjugate(e), 0, 0, 0], 
                    [0, conjugate(e), 0, 0]])
    return tau
    
def get_n_o(eta, theta, is_first):
    if is_first:
        return eta / 2.0 * (eye(4) + get_sigma(theta))
    return eta / 2.0 * (eye(4) + get_tau(theta))

def get_n_e(eta, theta, is_first):
    if is_first:
        return eta / 2.0 * (eye(4) - get_sigma(theta))
    return eta / 2.0 * (eye(4) - get_tau(theta))

def get_n_u(eta, theta, is_first):
    if is_first:
        return (1 - eta)
    return (1 - eta)
    
def get_B(eta, theta):
    A = eta / 2.0 * (E**(-2 * I * theta) - 1)
    B =  E**(-2 * I * theta) - 1
    tmp = 1 - eta
    M = eta / 2.0 * Matrix([[2 - eta, tmp, tmp, conjugate(A)*conjugate(B) - eta],\
    [tmp, 2 - eta, A*conjugate(B) - eta, tmp], [tmp, conjugate(A)*B - eta, 2 - eta, tmp],\
    [A*B - eta, tmp, tmp, 2 - eta]])
    return M
    
 # try to find minimum for psi = [psi1 psi2 psi3 psi4]   
def psi_opt(det, to_min):
    vars = [theta, psi1, psi2, psi3, psi4]
    sympy_2_numpy = sympy_to_numpy(tuple(vars))
    deriv_sympy_func = diff_sympy_func(vars)
    n_det = sympy_2_numpy(-re(det))
    deriv_sympy = deriv_sympy_func(-re(det))
    deriv_numpy = lambda_out_array(map(lambda x: function_wrapper(sympy_2_numpy(x), 1), deriv_sympy), 1)
    def func_to_min(args):
        return function_wrapper(sympy_2_numpy(to_min), 1)(args).real
    def deriv_func_to_min(args):        
        min_func = to_min
        res = lambda_out_array(map(lambda x: function_wrapper(sympy_2_numpy(x), 1), deriv_sympy_func(min_func)), 1)
        return res(args).real
    s = 0.5
    x0 =  [np.pi/3.0, s, s, s, s, s, s, s, s]
    psi_cons = complex_wrapper(lambda p : np.vdot(p[1:], p[1:]).real - 1, 1)
    psi_cons_deriv = array_stretcher(complex_wrapper(lambda x: [0, 2*abs(x[1]), 2*abs(x[2]), 2*abs(x[3]), 2*abs(x[4])], 1), 1)
    cons = (\
    {'type': 'ineq', 'fun' : function_wrapper(n_det, 1),\
    'jac' : deriv_numpy}, 
    {'type': 'eq', 'fun' : psi_cons,'jac' : psi_cons_deriv})
    res = minimize(func_to_min, x0, \
    jac=False,\
    constraints=cons, method='SLSQP', options={'disp': True})
    print res.x
    return res.x
    
# try to find minimum for psi = psi(r, omega)    
def r_omega_opt(det, to_min):
    vars = [theta, r, omega]
    sympy_2_numpy = sympy_to_numpy(tuple(vars))
    deriv_sympy_func = diff_sympy_func(vars)
    n_det = sympy_2_numpy(-re(det))
    deriv_sympy = deriv_sympy_func(-re(det))
    deriv_numpy = lambda_out_array(map(lambda x: function_wrapper(sympy_2_numpy(x), 3), deriv_sympy), 3)
    def func_to_min(args):
        return function_wrapper(sympy_2_numpy(to_min), 3)(args).real
    def deriv_func_to_min(args):        
        min_func = to_min
        res = lambda_out_array(map(lambda x: function_wrapper(sympy_2_numpy(x), 3), deriv_sympy_func(min_func)), 3)
        return res(args).real
    cons = (\
    {'type': 'ineq', 'fun' : function_wrapper(n_det, 3),\
    'jac' : deriv_numpy})
    bnds = ((0, 2*np.pi), (0, 1), (0, 2*np.pi))
    res = minimize(func_to_min, [0.0, 0.0001, 0.0], \
    jac=False,\
    constraints=cons, bounds=bnds, method='SLSQP', options={'disp': True})
    print res.x
    return func_to_min(res.x)
    
def ebernhard_opt(det, to_min):
    vars = [theta]
    sympy_2_numpy = sympy_to_numpy(tuple(vars))
    n_det = function_wrapper(sympy_2_numpy(re(det)), 1)
    res = minimize(n_det, [np.pi/3], \
    jac=False,  method='Nelder-Mead')
    print 'min_x = {}'.format(res.x / np.pi * 180)
    vars = [r, omega]
    sympy_2_numpy = sympy_to_numpy(tuple(vars))
    to_min_subs = to_min.subs([(theta, res.x)])
    #print to_min_subs
    plotting.plot3d(to_min_subs, (r, -1, 1), (omega, 0, 2*pi), \
    xlabel=r'values of $r$', ylabel=r'values of $\omega$', title=r'$J$')
    def func_to_min(args):
        return function_wrapper(sympy_2_numpy(to_min_subs), 2)(args)
    bnds = ((-1, 1), (0, 2*np.pi))
    res = minimize(func_to_min, [0, 0], \
    jac=False, bounds=bnds, method='SLSQP', options={'disp': True})
    print 'res = {}'.format(res.x)
    return res.x

    
def eta_K_opt(det, to_min):
    etas =  [0.667] + [0.01*x for x in range(70, 81, 5)]
    res = []
    for e1 in etas:
        #res2 = []
        #for e2 in etas:
        to_subs = [(eta1, e1), (eta2, e1)]
        print e1
        subs_det = det.subs(to_subs)
        #plotting.plot(re(subs_det), (theta, -0.008*pi, 0.008*pi), \
        #xlabel=r'values of $\theta$', title=r'$\det\mathcal{B}$')
        subs_to_min = to_min.subs(to_subs)
        res.append(ebernhard_opt(subs_det, subs_to_min))
        #res.append(res2)
    print res
    return res
    

init_printing()
eta = Symbol('eta', integer=True)
x = Symbol('x', integer=True)
#eta1, eta2 = symbols('eta_1 eta_2', integer=True)
eta1 = eta
eta2 = eta 
ksi = Symbol('ksi', integer=True)

alpha1, alpha2, beta1, beta2 = symbols('alpha_1 alpha_2 beta_1, beta_2',integer=True )
theta = Symbol('theta', integer=True)
n_oo = lambda alpha, beta : get_n_o(eta1, alpha, True).multiply( \
get_n_o(eta2, beta, False))
n_oe = lambda alpha, beta : get_n_o(eta1, alpha, True).multiply( \
get_n_e(eta2, beta, False))
n_ou = lambda alpha, beta : get_n_o(eta1, alpha, True) * \
get_n_u(eta2, beta, False)
n_eo = lambda alpha, beta : get_n_e(eta1, alpha, True).multiply( \
get_n_o(eta2, beta, False))
n_uo = lambda alpha, beta : get_n_u(eta1, alpha, True) * \
get_n_o(eta2, beta, False)
n = n_oe(0, theta) + n_ou(0, theta) + n_eo(theta, 0) + \
n_uo(theta, 0) + n_oo(theta, theta) - n_oo(0, 0)

psi1, psi2, psi3, psi4 = symbols('psi_1 psi_2 psi_3 psi_4')
r, omega = symbols('r omega', integer=True)
#psi = Matrix([psi1, psi2, psi3, psi4])
psi = 1 / 2.0 / sqrt(1 + r**2) * Matrix([(1+r)*E**(-I*omega), -(1-r), -(1-r), (1+r)*E**(I*omega)])
quant_mean = psi.T * n * psi 
quant_std = psi.T * n * n * psi - quant_mean*quant_mean
#print 'mean = {}'.format(latex(quant_mean[0]))
#print 'std = {}'.format(latex(quant_std))

#to_min = quant_mean[0]**2 / quant_std[0]
#to_min = quant_mean[0] / sqrt(quant_std[0])
# [0] is because the resut of matrix multiplication is always matrix here
#opt_val = eta_K_opt(n.det(), to_min, quant_mean[0])
to_min = quant_mean[0]
opt_val = eta_K_opt(n.det(), to_min)
