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
    e = E**(2*I*theta)
    sigma = Matrix([[0, e, 0, 0], [conjugate(e), 0, 0, 0], [0, 0, 0, e], 
                    [0, 0, conjugate(e), 0]])
    return sigma
    
def get_tau(theta):
    e = E**(2*I*theta)
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
    
def eigenvalues_analyse():
    eigenvalues = [k for (k, v) in n.eigenvals().iteritems()]
    #eigenvectors = n.eigenvects()
    #print eigenvectors
    #print 'eigenvalues: {}'.format(latex(map(lambda (k,v): k, n.eigenvals().iteritems())))
    #p1 = plotting.plot3d(eigenvalues[0], (eta, 0.01, 1), (theta, 0, 2*pi), \
    #xlabel=r'values of $\eta$',ylabel=r'values of $\theta$', title=r'$\lambda_1$')
    #p2 = plotting.plot3d(eigenvalues[1], (eta, 0.01, 1), (theta, 0, 2*pi), \
    #xlabel=r'values of $\eta$',ylabel=r'values of $\theta$', title=r'$\lambda_2$')
    #p3 = plotting.plot3d(eigenvalues[2], (eta, 0.01, 1), (theta, 0, 2*pi), \
    #xlabel=r'values of $\eta$',ylabel=r'values of $\theta$', title=r'$\lambda_3$')
    #p4 = plotting.plot3d(eigenvalues[3], (eta, 0.01, 1), (theta, 0, 2*pi), \
    #xlabel=r'values of $\eta$',ylabel=r'values of $\theta$', title=r'$\lambda_4$')
    return eigenvalues

    
def min_eta():
    t1s = [0.01*pi*x for x in range(0, 200)]
    t2s = [0.01*x for x in range(85, 100)]
    min_eigenvals = []
    #max_eigenvals = []
    for t2 in t2s:
        min_vals = []
     #   max_vals = []
        for t1 in t1s:
            if re(n.det().evalf(subs={theta: t1, eta: t2})) < 0:
                es = [re(eigenvalues[i].evalf(subs={theta: t1, eta: t2})) for i in range(4)]
                min_vals.append((min(es), es.index(min(es))))
     #           max_neg = 0
     #           for e in es:
     #               if e < max_neg:
     #                   max_neg = e
     #           max_vals.append(max_neg)
        all_mins = [f for (f, s) in min_vals]
        if min_vals == []:
            min_eigenvals.extend([1])
            continue
        min_eigenvals.extend([min(all_mins)])
     #   max_eigenvals.append(min(max_vals))
    pl.plot(t2s, min_eigenvals)
    #pl.plot(t2s, max_eigenvals)
    pl.show()
    print min_eigenvals
    
def all_opt_min(det, eigen):
    vars = [eta1, eta2, theta]
    sympy_2_numpy = sympy_to_numpy(tuple(vars))
    deriv_sympy_func = diff_sympy_func(vars)
    n_det = sympy_2_numpy(-re(det))
    deriv_sympy = deriv_sympy_func(-re(det))
    deriv_numpy = lambda_out_array(map(lambda x: function_wrapper(sympy_2_numpy(x)), deriv_sympy))
    def func_to_min(args):
        eigen_numpy = map(lambda x: function_wrapper(sympy_2_numpy(x)), eigen)
        return min(map(lambda f: f(args).real, eigen_numpy))
    def deriv_func_to_min(args):        
        eigen_numpy = map(lambda x: function_wrapper(sympy_2_numpy(x)), eigen)
        es = map(lambda f: f(args), eigen_numpy)
        min_index = es.index(min(es))
        min_func = eigen[min_index]
        res = lambda_out_array(map(lambda x: function_wrapper(sympy_2_numpy(x)), deriv_sympy_func(min_func)))
        return res(args).real
    cons = ({'type': 'ineq', 'fun' : function_wrapper(n_det),\
    'jac' : deriv_numpy})
    bnds = ((0, 1), (0, 1), (0, 2*np.pi))    
    res = minimize(func_to_min, [0.75, 0.75, np.pi/3.0], jac=deriv_func_to_min,\
    constraints=cons, bounds=bnds, method='SLSQP', options={'disp': True})
    print res.x
    c = function_wrapper(n_det)
    print c(res.x)
    
def theta_opt_min(det, eigen, val1, val2, is_eta_diff=False):
    to_subs = []
    if is_eta_diff:
        to_subs = [(eta, val1), (x, val2)]
    else:
        to_subs = [(eta1, val1), (eta2, val2)]
    det = det().subs(to_subs)
    eigen = map(lambda expr: expr.subs(to_subs), eigen)
    vars = [theta]
    sympy_2_numpy = sympy_to_numpy(tuple(vars))
    deriv_sympy_func = diff_sympy_func(vars)
    n_det = sympy_2_numpy(-re(det))
    deriv_sympy = deriv_sympy_func(-re(det))
    deriv_numpy = lambda_out_array(map(lambda x: sympy_2_numpy(x), deriv_sympy))
    def func_to_min(args):
        eigen_numpy = map(lambda x: sympy_2_numpy(x), eigen)
        return min(map(lambda f: f(args).real, eigen_numpy))
    def deriv_func_to_min(args):        
        eigen_numpy = map(lambda x: sympy_2_numpy(x), eigen)
        es = map(lambda f: f(args), eigen_numpy)
        min_index = es.index(min(es))
        min_func = eigen[min_index]
        res = lambda_out_array(map(lambda x: sympy_2_numpy(x), deriv_sympy_func(min_func)))
        return res(args).real
    cons = ({'type': 'ineq', 'fun' : n_det, 'jac' : deriv_numpy})
    bnds = ((0, 2.0*np.pi),)
    res = minimize(func_to_min, (np.pi/3.0), jac=deriv_func_to_min,\
    constraints=cons, bounds=bnds, method='SLSQP', options={'disp': True})
    print res.x
    return func_to_min(res.x)
    
def eta_2nd_level_min(det, eigen):
    return [theta_opt_min(det, eigen, 0.01*val, 0.01*val) for val in range(60, 101, 2)]
    
def different_eta_2nd_level_min(det, eigen, eta_val):
    first = np.int(-100*(eta_val - 0.6) - 0.5)
    second = np.int(100*(1 - eta_val) + 0.5)
    print first
    print second
    return [theta_opt_min(det, eigen, eta_val, 0.01*val, True) for val in range(first, second+1, 5)]
    
def different_eta_3rd_level_min(det, eigen):
    res = [different_eta_2nd_level_min(det, eigen, 0.01*val) for val in range(60, 101, 5)]
    print res
    return map(lambda r: (min(r), r.index(min(r))), res)
    
def psi_opt_min(det, to_min, mean_cons):
    own_function_wrapper = lambda f: function_wrapper(f)
    vars = [eta, theta, psi1, psi2, psi3, psi4]
    sympy_2_numpy = sympy_to_numpy(tuple(vars))
    deriv_sympy_func = diff_sympy_func(vars)
    n_det = sympy_2_numpy(-re(det))
    deriv_sympy = deriv_sympy_func(-re(det))
    numpy_mean_cons = own_function_wrapper(sympy_2_numpy(-mean_cons))
    deriv_mean_cons_sympy = deriv_sympy_func(-mean_cons)
    deriv_mean_cons_numpy = lambda_out_array(map(lambda x: own_function_wrapper(sympy_2_numpy(x)), deriv_mean_cons_sympy))
    def wrapper(f, N):
        def helper():
             n = len(arg)
             args = [arg[i] for i in range(n-N)]
             args.append(np.matrix([arg[i] for i in range(n-N+1, n)]))
             return f(*(args))
        return lambda f: (lambda arg: helper())
    deriv_numpy = lambda_out_array(map(lambda x: own_function_wrapper(sympy_2_numpy(x)), deriv_sympy))
    def func_to_min(args):
        return own_function_wrapper(sympy_2_numpy(to_min))(args).real
    def deriv_func_to_min(args):        
        min_func = to_min
        res = lambda_out_array(map(lambda x: own_function_wrapper(sympy_2_numpy(x)), deriv_sympy_func(min_func)))
        return res(args).real
    psi_cons = lambda p : np.dot(p[2:], p[2:]) - 1
    cons = ({'type': 'ineq', 'fun' : numpy_mean_cons,\
    'jac' : deriv_mean_cons_numpy}, 
    {'type': 'ineq', 'fun' : own_function_wrapper(n_det),\
    'jac' : deriv_numpy}, 
    {'type': 'eq', 'fun' : psi_cons,'jac' : lambda x: [0, 0, 2*x[2], 2*x[3], 2*x[4], 2*x[5]]})
    bnds = ((0.7, 0.8), (0, 2*np.pi), (0, 1), (0, 1), (0, 1), (0, 1))    
    res = minimize(func_to_min, [0.8, np.pi/3.0, 0.5, 0.5, 0.5, 0.5], \
    jac=False,\
    constraints=cons, bounds=bnds, method='SLSQP', options={'disp': True})
    print res.x
    return res.x
    
    
def K_opt_min(det, to_min, mean_cons):
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
        return array_stretcher(res(args).real)
    x0 =  [np.pi/3.0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5]
    psi_cons = complex_wrapper(lambda p : np.vdot(p[1:], p[1:]).real - 1, 1)
    psi_cons_deriv = array_stretcher(complex_wrapper(lambda x: [0, 2*abs(x[1]), 2*abs(x[2]), 2*abs(x[3]), 2*abs(x[4])], 1), 1)
    print len(complex_array_wrapper(x0, 1))
    cons = (\
    {'type': 'ineq', 'fun' : function_wrapper(n_det, 1),\
    'jac' : deriv_numpy}, 
    {'type': 'eq', 'fun' : psi_cons,'jac' : psi_cons_deriv})
    res = minimize(func_to_min, x0, \
    jac=False,\
    constraints=cons, method='SLSQP', options={'disp': True})
    print res.x
    return res.x#-function_wrapper(n_det)(res.x)
    
    
def K_opt_min2(det, to_min, mean_cons):
    vars = [theta, r, omega]
    sympy_2_numpy = sympy_to_numpy(tuple(vars))
    deriv_sympy_func = diff_sympy_func(vars)
    n_det = sympy_2_numpy(-re(det))
    deriv_sympy = deriv_sympy_func(-re(det))
    deriv_numpy = lambda_out_array(map(lambda x: function_wrapper(sympy_2_numpy(x)), deriv_sympy))
    numpy_mean_cons = function_wrapper(sympy_2_numpy(-mean_cons))
    deriv_mean_cons_sympy = deriv_sympy_func(-mean_cons)
    deriv_mean_cons_numpy = lambda_out_array(map(lambda x: function_wrapper(sympy_2_numpy(x)), deriv_mean_cons_sympy))    
    def func_to_min(args):
        return function_wrapper(sympy_2_numpy(to_min))(args).real
    def deriv_func_to_min(args):        
        min_func = to_min
        res = lambda_out_array(map(lambda x: function_wrapper(sympy_2_numpy(x)), deriv_sympy_func(min_func)))
        return res(args).real
    cons = (\
    {'type': 'ineq', 'fun' : function_wrapper(n_det),\
    'jac' : deriv_numpy})
    bnds = ((0, 2*np.pi), (0, 1), (0, 2*np.pi))
    res = minimize(func_to_min, [np.pi/3.0, 0.001, 0], \
    jac=False,\
    constraints=cons, bounds=bnds, method='SLSQP', options={'disp': True})
    print res.x
    return func_to_min(res.x)
    
def eta_K_opt(det, to_min, mean_cons):
    etas =  [0.01*x for x in range(60, 81, 2)]
    res = []
    for e1 in etas:
        #res2 = []
        #for e2 in etas:
        to_subs = [(eta1, e1), (eta2, e1)]
        subs_det = det.subs(to_subs)
        subs_to_min = to_min.subs(to_subs)
        subs_mean_cons = mean_cons.subs(to_subs)
        res.append(K_opt_min(subs_det, subs_to_min, subs_mean_cons))
        #res.append(res2)
    print res
    return res
    
init_printing()
eta = Symbol('eta', integer=True)
x = Symbol('x', integer=True)
#eta1, eta2 = symbols('eta_1 eta_2', integer=True)
eta1 = eta
eta2 = eta 
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
n_fun = lambdify(theta, n.det(), 'numpy')
det_fun = lambdify(theta, n.det(), 'sympy')

eigenvalues = eigenvalues_analyse()

psi1, psi2, psi3, psi4 = symbols('psi_1 psi_2 psi_3 psi_4')
r, omega = symbols('r omega')
psi = Matrix([psi1, psi2, psi3, psi4])
#psi = 1 / 2.0 / sqrt(1 + r**2) * Matrix([(1+r)*E**(-I*omega), -(1-r), -(1-r), (1+r)*E**(I*omega)])
quant_mean = psi.T * n * psi
quant_std = psi.T * n * n * psi - quant_mean*quant_mean
#to_min = -quant_mean**2 + quant_std
#to_min = sqrt(re(quant_std[0])) - quant_mean[0]
#to_min = quant_mean[0]**2 / quant_std[0]
to_min = quant_mean[0] / sqrt(quant_std[0])
# [0] is because the resut of matrix multiplication is always matrix here
opt_val = eta_K_opt(n.det(), to_min, quant_mean[0])
to_min = quant_mean[0]
opt_val = eta_K_opt(n.det(), to_min, quant_mean[0])
