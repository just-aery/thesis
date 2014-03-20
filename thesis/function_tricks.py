# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:49:25 2013

@author: polina

Some used tricks to convert from symbolic expressions to multivariable numeric functions
"""

from sympy import lambdify, diff
from sympy.printing.lambdarepr import LambdaPrinter 
import numpy as np

class ImaginaryPrinter(LambdaPrinter):
     def _print_ImaginaryUnit(self, expr):
         return "1j"
     pass

#creates numeric function from symbolic expression
def sympy_to_numpy(variables):
    return lambda expr: lambdify(variables, expr, modules='numpy', printer=ImaginaryPrinter)

#adds zero after every number in result array starting from index #from_i
#it is used to add zeros in array of gradients if we want to represent function with n complex arguments as a function 
# with 2n real arguments
def array_stretcher_helper(arr, from_i):
    res = []
    c = 0
    for i in arr:
        res.append(i)
        if c >= from_i:
            res.append(0)
        c += 1
    return np.array(res)
 
#function that adds zeros to result after f(arg) computation   
array_stretcher = lambda f, from_i: (lambda arg: array_stretcher_helper(f(arg), from_i))

#creates function that returns array from array of functions
def lambda_out_array_helper(expr):
    return lambda arg: np.array(map(lambda e: e(arg), expr))

#creates function that returns array from array of functions with additional zeros for complex parameters
lambda_out_array = lambda arg, from_i: array_stretcher(lambda_out_array_helper(arg), from_i)
    
#agr - nx1 vector of fumction arguments
# calls function with n arguments: arg=[a, b, c] -> f(a, b, c)
def array_to_args(f, arg):
    n = len(arg)
    args = [arg[i] for i in range(n)]
    return f(*(args)).real
    
#creates n complex parameters from 2n real numers in array from index #from_i
def complex_array_wrapper(arg, from_i):
    a = arg[0:from_i]
    b = [arg[i] + arg[i+1]*1j for i in range(from_i, len(arg), 2)]
    return  np.append(a, b)
 
#given function f - function with n complex parameters f(a1, a2, ... an)
# allows to call f(arg) where arg - vector with real values (complex values are represented as [... re, im, ...])   
function_wrapper = lambda f, from_i: (lambda arg: array_to_args(f, complex_array_wrapper(arg, from_i)))

#allows call function with n complex parameters to be called with 2n real parameters instead
complex_wrapper = lambda f, from_i: (lambda arg: f(complex_array_wrapper(arg, from_i)))

#creates symbolic array of derivatives of symbolic function
def diff_sympy_func(variables):
    return lambda expr: map(lambda var: diff(expr, var), variables)