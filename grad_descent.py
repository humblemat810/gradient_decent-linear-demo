# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 19:00:52 2022
Task Description
Write a function that given a set of points P for a 2D space (described as tuples Pi = (Xi, Yi)), 
performs a search for the best fit of a linear function of the form y=m*x+b using *gradient descent algorithm*. 
The search goal is the minimal MSE. 
MSE = 1/n * SUM((Yi-Ei)**2), where Yi are the actual values and Ei are the estimated values.

The outline of the gradient descent algorithm is as follows:
1. Initial approximation can be selected at your discretion (m_0, b_0)
2. The goal function F(m, b) is MSE function described above
3. At each step (m_n+1, b_n+1) is calculated based on the previous approximation as 
    (m_n+1, b_n+1) = (m_n, b_n) - L * \grad(F(m_n, b_n)), 
    where \grad is the gradient of function F, 
    and 0 < L < 1 is a constant learning rate.
4. The search ends either when the number of steps n reaches a limit, 
    or when the MSE becomes sufficiently small

Test case 1:
points = [[0,0], [1,1], [1.9,2], [3,3.2], [4,4.1], [5,5.11]]

Test case 2: 
points2 = [[0,0], [1,1], [2,2], [3,3], [4,4], [5,5]]
@author: chanh
"""

import sympy as sym

m = sym.Symbol('m')
b = sym.Symbol('b')

y = sym.Symbol('y')
x = sym.Symbol('x')

grad_m_sym = sym.diff((y - (m*x + b))**2, m)
grad_b_sym = sym.diff((y - (m*x + b))**2, b)

print(grad_m_sym)
print(grad_b_sym)

import numpy as np

max_iteration = 200
l = 0.1
epsilon = 0.01

def mse(points, m, b):
    total = 0
    for pt in points:
        x, y = pt
        total += (y - (x * m + b)) ** 2
    return total / len(points)       
    

def gradient(points, m, b):
    "(y - (mx + b))**2    2 * delta y * (x + 1) "
    grad_b = 0
    grad_m = 0
    n = len(points)
    for pt in points :
        x, y = pt
        grad_m += 2 * (y - m * x - b) * (x)
        grad_b += 2 * (y - m * x - b) 
    return grad_m / n, grad_b / n

def main(points, m = None, b = None):
    points = np.array(points)
    mean_pt = points.mean(axis=0)
    if m is None:
        m = mean_pt[1] / mean_pt[0]
    if b is None:
        b = 0
        
    i = 0
    
    while i < max_iteration:
        grad_tuple = gradient(points, m, b)
        
        grad_m, grad_b = gradient(points, m, b)
        m += l * grad_m
        b += l * grad_b
        err = mse(points, m, b)
        if i % 10 == 0:
            print("gradient_tuple = ", grad_tuple)
            print('m = ', m, "b = ", b)
            print('sum MSE error = ', err)
        if err < epsilon:
            return m, b
        i += 1
    return m, b
    
    



if __name__ == "__main__":
    
    # test_points = [[0,0], [[1,1], [1.9,2], [3,3.2], [4,4.1], [5,5.11]]
    test_points = [[0,0], [1,1]]
    main(test_points, -1, 1)