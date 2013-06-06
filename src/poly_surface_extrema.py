#!/usr/bin/env python
"""
@author: Benjamin Chretien
"""
import math
import numpy as np
from mayavi import mlab

j = complex(0,1)
min_x = -10.
max_x =  10.
min_y = -8.
max_y =  8.
root0 = 1.
lamda = 0.01/abs(max_x)
step_size = 0.1

def f_evolution_element(x, y):
  root_real = 2.
  roots = np.zeros((3,3))
  if y < 0:
    dP = np.poly([root0, root_real + y * j, root_real - y * j])
  elif y > 0:
    dP = np.poly([root0, root_real+y, root_real-y])
  else:
    dP = np.poly([root0, root_real, -root_real])

  P = lamda*np.polyint(dP)
  cplx_roots = np.roots(dP)
  roots[:,0] = [_.real for _ in cplx_roots if _.real < max_x and _.real > min_x]
  roots[:,0] = np.sort(roots[:,0])
  z = np.polyval(P, x)
  for i in xrange(roots.shape[0]):
    roots[i,1] = y
    roots[i,2] = np.polyval(P, roots[i,0])
  return z,roots

def f_evolution(x, y):
  z = np.zeros((x.size, y.size))
  root_real = 2.
  roots = np.zeros((3,y.size,3))
  for k in xrange(y.size):
    if y[k] < 0:
      dP = np.poly([root0, root_real + y[k] * j, root_real - y[k] * j])
    elif y[k] > 0:
      dP = np.poly([root0, root_real + y[k], root_real-y[k]])
    else:
      dP = np.poly([root0, root_real, -root_real])

    P = lamda*np.polyint(dP)
    cplx_roots = np.roots(dP)
    roots[:,k,0] = [_.real for _ in cplx_roots if _.real < max_x and _.real > min_x]
    roots[:,k,0] = np.sort(roots[:,k,0])
    for i in xrange(x.size):
      z[i,k] = np.polyval(P, x[i])
    for i in xrange(roots.shape[0]):
      roots[i,k,1] = y[k]
      roots[i,k,2] = np.polyval(P, roots[i,k,0])
  return z,roots

# Grid
X = np.arange(min_x, max_x + step_size, step_size)
Y = np.arange(min_y, max_y + step_size, step_size)

# Compute data
Z_evol,roots_evol = f_evolution(X,Y)

fig = mlab.figure('Complex roots', bgcolor=(0, 0, 0), size=(800, 600))

# Clamp colors to get a better gradient near the minimum
vmin_1 = np.min(Z_evol[:,0:10])
vmax_1 = vmin_1 + 0.02*(np.max(Z_evol[:,0:10]) - vmin_1)

# Create the surface
s_poly = mlab.surf(X[:],Y[:],Z_evol[:,:], colormap='jet',
                   representation='surface',
                   vmin = vmin_1, vmax = vmax_1,
                   figure=fig)

# Real root
x = roots_evol[0,0:math.floor(len(Y)/2)+1,0].flatten(0)
y = roots_evol[0,0:math.floor(len(Y)/2)+1,1].flatten(0)
z = roots_evol[0,0:math.floor(len(Y)/2)+1,2].flatten(0)
trajectory1 = mlab.plot3d(x[:], y[:], z[:],
                           color=(1,0,0), tube_radius=None)

# Real part of conjugate root
x = roots_evol[2,0:math.floor(len(Y)/2)+1,0].flatten(0)
y = roots_evol[2,0:math.floor(len(Y)/2)+1,1].flatten(0)
z = roots_evol[2,0:math.floor(len(Y)/2)+1,2].flatten(0)
trajectory2 = mlab.plot3d(x[:], y[:], z[:],
                           color=(1,1,0), tube_radius=None)

# Real root
x = roots_evol[2,math.floor(len(Y)/2):-1,0].flatten(0)
y = roots_evol[2,math.floor(len(Y)/2):-1,1].flatten(0)
z = roots_evol[2,math.floor(len(Y)/2):-1,2].flatten(0)
trajectory3 = mlab.plot3d(x[:], y[:], z[:],
                           color=(1,1,0), tube_radius=None)

# Real root
x = roots_evol[0,math.floor(len(Y)/2):-1,0].flatten(0)
y = roots_evol[0,math.floor(len(Y)/2):-1,1].flatten(0)
z = roots_evol[0,math.floor(len(Y)/2):-1,2].flatten(0)
trajectory4 = mlab.plot3d(x[:], y[:], z[:],
                           color=(0,1,0), tube_radius=None)

# Real root
x = roots_evol[1,math.floor(len(Y)/2):-1,0].flatten(0)
y = roots_evol[1,math.floor(len(Y)/2):-1,1].flatten(0)
z = roots_evol[1,math.floor(len(Y)/2):-1,2].flatten(0)
trajectory5 = mlab.plot3d(x[:], y[:], z[:],
                           color=(1,0,0), tube_radius=None)

# Create the axes
mlab.axes(s_poly, color=(.7, .7, .7),
          xlabel='x', ylabel='y < 0: Imag(conj_root)\ny > 0: +/- real root', zlabel='P(x)')

# Show the result
mlab.show()
