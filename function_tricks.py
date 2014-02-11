# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:49:25 2013

@author: polina

Some used tricks to convert from symbolic expressions to multivariable functions
"""

from sympy import lambdify, diff
from sympy.printing.lambdarepr import LambdaPrinter 
import numpy as np

class ImaginaryPrinter(LambdaPrinter):
     def _print_ImaginaryUnit(self, expr):
         return "1j"
     pass


def sympy_to_numpy(variables):
    return lambda expr: lambdify(variables, expr, modules='numpy', printer=ImaginaryPrinter)

def array_stretcher_helper(arr):
    res = []
    for i in arr:
        res.append(i)
        res.append(0)
    return np.array(res)
    
array_stretcher = lambda f: (lambda arg: array_stretcher_helper(f(arg)))

def lambda_out_array_helper(expr):
    return lambda arg: np.array(map(lambda e: e(arg), expr))

lambda_out_array = lambda arg: array_stretcher(lambda_out_array_helper(arg))
    
def array_to_args(f, arg):
    n = len(arg)
    args = [arg[i] for i in range(n)]
    return f(*(args)).real
    
complex_array_wrapper = lambda arg: [arg[i] + arg[i+1]*1j for i in range(0, len(arg), 2)]
    
function_wrapper = lambda f: (lambda arg: array_to_args(f, complex_array_wrapper(arg)))

complex_wrapper = lambda f: (lambda arg: f(complex_array_wrapper(arg)))




def diff_sympy_func(variables):
    return lambda expr: map(lambda var: diff(expr, var), variables)