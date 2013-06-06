#!/usr/bin/env python
"""
@author: Benjamin Chrétien <chretien@lirmm.fr>
"""
import numpy as np
import math
import sys

def cubic_roots(poly):
  """
      See: https://www.e-education.psu.edu/png520/m11_p6.html
  """
  epsilon = sys.float_info.epsilon
  if len(poly) != 4 or abs(poly[0]) < epsilon:
    print('Wrong order')
    return []

  a = poly[1]/poly[0]
  b = poly[2]/poly[0]
  c = poly[3]/poly[0]

  Q = (a*a - 3.0*b)/9.0
  R = (2.0*a*a*a - 9.0*a*b + 27.0*c)/54.0
  M = R*R - Q*Q*Q

  roots = []

  # 1 real root
  if M > 100*epsilon:
    S = -math.copysign((abs(R) + math.sqrt(M))**(1.0/3.0), R)
    U = 0
    if abs(S) > epsilon:
      U = Q/S
    roots.append(S + U - a/3.0)
  # 3 real roots
  elif Q*Q*Q > epsilon: #if M < -epsilon
    theta = math.acos(R/(math.sqrt(Q*Q*Q)))
    roots.append(-2.0*math.sqrt(Q)*math.cos(theta/3.0) - a/3.0)
    roots.append(-2.0*math.sqrt(Q)*math.cos((theta + 2*math.pi)/3.0) - a/3.0)
    roots.append(-2.0*math.sqrt(Q)*math.cos((theta - 2*math.pi)/3.0) - a/3.0)
  # 3 real roots
  else:
    roots.append(-2.0*math.sqrt(Q) - a/3.0)
    roots.append(math.sqrt(Q) - a/3.0)
    roots.append(math.sqrt(Q) - a/3.0)

  return roots

def quartic_roots(poly):
  """
      See: http://mathworld.wolfram.com/QuarticEquation.html
  """
  epsilon = sys.float_info.epsilon

  if len(poly) != 5 or abs(poly[0]) < epsilon:
    print('Wrong order')
    return []

  a3 = poly[1]/poly[0]
  a2 = poly[2]/poly[0]
  a1 = poly[3]/poly[0]
  a0 = poly[4]/poly[0]

  # Create the resolvent polynomial
  resolvent_poly = [1.0,
                    -a2,
                    a1*a3 - 4.0*a0,
                    4.0*a2*a0 - a1*a1 - a3*a3*a0]

  # Compute the roots of the resolvent polynomial
  resolvent_roots = cubic_roots(resolvent_poly)

  # Choose one of the roots such as R^2 > 0, if possible
  y1 = resolvent_roots[0]
  square_R = 0.25*a3*a3 - a2 + y1

  if square_R < 100*epsilon:
    for r in resolvent_roots[1:]:
      y1 = r
      square_R = 0.25*a3*a3 - a2 + y1
      if square_R >= epsilon:
        break


  roots = []

  if square_R >= 0:
    square_D = 0
    square_E = 0
    R = math.sqrt(square_R)
    if abs(R) > epsilon:
      square_D = 0.75*a3*a3 -square_R -2.0*a2 + 0.25*(4.0*a3*a2 -8.0*a1 - a3*a3*a3)/R
      square_E = 0.75*a3*a3 -square_R -2.0*a2 - 0.25*(4.0*a3*a2 -8.0*a1 - a3*a3*a3)/R
    elif y1*y1 -4.0*a0 >= 0:
      square_D = 0.75*a3*a3 -2.0*a2 + 2.0*math.sqrt(y1*y1 -4.0*a0)
      square_E = 0.75*a3*a3 -2.0*a2 - 2.0*math.sqrt(y1*y1 -4.0*a0)
    else:
      # R = 0, D and E complex => 4 complex roots
      return roots

    if square_D >= 0:
      D = math.sqrt(square_D)
      roots.append(-0.25*a3 + 0.5*(R + D))
      roots.append(-0.25*a3 + 0.5*(R - D))

    if square_E >= 0:
      E = math.sqrt(square_E)
      roots.append(-0.25*a3 - 0.5*(R - E))
      roots.append(-0.25*a3 - 0.5*(R + E))

    return roots

  else:
    # R complex => 4 complex roots
    return roots


j = complex(0,1)
print('############################ CUBIC ###################################')
# From roots
roots = np.array([[1, 1, 1], [1, 2, 3], [1, 3, 3],
                  [1, j, -j], [-5, 3*j, -3*j]]);
for r in roots:
  sorted_r = np.sort(r)
  filtered_r = [_.real for _ in r if (not isinstance(_, complex))
                                  or (np.abs(_.imag) < 1e-12)]
  test_poly = np.poly(sorted_r)
  np_roots = np.roots(test_poly)
  np_roots = [_.real for _ in np_roots if (not isinstance(_, complex))
                                       or (np.abs(_.imag) < 1e-12)]
  np_roots_sorted = np.sort(np_roots)
  print 'Numpy roots: ', np_roots_sorted
  analytical_roots = cubic_roots(test_poly)
  analytical_roots_sorted = np.sort(analytical_roots)
  print 'Analytical roots: ', analytical_roots_sorted
  assert(np.allclose(filtered_r, analytical_roots_sorted,
                     rtol=1e-05, atol=1e-08))
  print

# from coeffs
coeffs = np.array([[1, 1, 1, 1], [1, 2, 3, 4], [1, 3, 3, 4], [1, 0, 0, 0],
                   [1, 0, 0, 1], [1, -6, 11, -6], [1, 7, 49, 343],
                   [1.0, -3, -12.0, 24.0], [1.0, -11.0, 32.0, -28.0],
                   [1.0, 5.0, -16.0, -80.0]]);
for c in coeffs:
  test_poly = np.poly1d(c)
  np_roots = np.roots(test_poly.coeffs)
  np_roots = [_ for _ in np_roots if (not isinstance(_, complex))
                                  or (np.abs(_.imag) < 1e-12)]
  np_roots_sorted = np.sort(np_roots)
  print 'Numpy roots: ', np_roots_sorted
  analytical_roots = cubic_roots(test_poly.coeffs)
  analytical_roots_sorted = np.sort(analytical_roots)
  print 'Analytical roots: ', analytical_roots_sorted
  assert(np.allclose(np_roots_sorted, analytical_roots_sorted,
                     rtol=1e-05, atol=1e-08))
  print

print('########################### QUARTIC ##################################')
# From roots
roots = np.array([[-2,-1,1,2], [1, 1, 1, 1], [1, 2, 3, 4], [1, 3, 3, 2],
                  [1, j, -j, -1], [-5*j, 5*j, -3*j, 3*j]]);
for r in roots:
  sorted_r = np.sort(r)
  filtered_r = [_.real for _ in r if (not isinstance(_, complex))
                                  or (np.abs(_.imag) < 1e-6)]
  filtered_r = np.sort(filtered_r)
  test_poly = np.poly(sorted_r)
  test_poly = test_poly.real
  print test_poly
  np_roots = np.roots(test_poly)
  np_roots = [_.real for _ in np_roots if (not isinstance(_, complex))
                                       or (np.abs(_.imag) < 1e-6)]
  np_roots_sorted = np.sort(np_roots)
  print 'Actual roots: ', filtered_r
  print 'Numpy roots: ', np_roots_sorted
  analytical_roots = quartic_roots(test_poly)
  analytical_roots_sorted = np.sort(analytical_roots)
  print 'Analytical roots: ', analytical_roots_sorted
  assert(np.allclose(filtered_r, analytical_roots_sorted,
                     rtol=1e-05, atol=1e-08))
  print

# from coeffs
coeffs = np.array([[1, 0, 2, 0, 1], [1, -3, 3, -3, 2], [1, 0, -5, 0, 4],
                   [1, 1, 1, 1, 1], [1, 2, 3, 4, 5],
                   [1, 3, 3, 4, 4], [1, 0, 0, 0, 0], [1, 0, 0, 0, 1],
                   [1, -6, 11, -6, 1], [1, 7, 49, 343, 1],
                   [1.0, -3.0, -12.0, 24.0, 1.0]]);
for c in coeffs:
  test_poly = np.poly1d(c)
  print test_poly
  np_roots = np.roots(test_poly.coeffs)
  np_roots = [_.real for _ in np_roots if (not isinstance(_, complex))
                                       or (np.abs(_.imag) < 1e-6)]
  np_roots_sorted = np.sort(np_roots)
  print 'Numpy roots: ', np_roots_sorted
  analytical_roots = quartic_roots(test_poly.coeffs)
  analytical_roots_sorted = np.sort(analytical_roots)
  print 'Analytical roots: ', analytical_roots_sorted
  assert(np.allclose(np_roots_sorted, analytical_roots_sorted,
                     rtol=1e-05, atol=1e-08))
  print