#!/usr/bin/env python
"""
@author: Benjamin Chretien <chretien@lirmm.fr>
"""
import numpy as np
import sympy as sp
import math
import sys

def cubic_roots(poly, return_conjugate_realpart = False):
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

    # Also return the real part of the complex conjugate roots
    if return_conjugate_realpart:
      roots.append(-0.5*(S + U) - a/3.0)
      roots.append(-0.5*(S + U) - a/3.0)

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

def quartic_roots(poly, return_conjugate_realpart = False):
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

  if square_R < 10*epsilon:
    for r in resolvent_roots[1:]:
      y1 = r
      square_R = 0.25*a3*a3 - a2 + y1
      if square_R >= epsilon:
        break

  roots = []
  R = 0

  # R >= 0
  if square_R >= -epsilon:
    square_D = 0
    square_E = 0

    # R > 0
    if abs(square_R) > epsilon:
      R = math.sqrt(abs(square_R))
      square_D = 0.75*a3*a3 -square_R -2.0*a2 + 0.25*(4.0*a3*a2 -8.0*a1 - a3*a3*a3)/R
      square_E = 0.75*a3*a3 -square_R -2.0*a2 - 0.25*(4.0*a3*a2 -8.0*a1 - a3*a3*a3)/R
    # R = 0, D^2 and E^2 real
    elif y1*y1 -4.0*a0 >= 0:
      square_D = 0.75*a3*a3 -2.0*a2 + 2.0*math.sqrt(y1*y1 -4.0*a0)
      square_E = 0.75*a3*a3 -2.0*a2 - 2.0*math.sqrt(y1*y1 -4.0*a0)
    # R = 0, D^2 and E^2 complex => 4 complex conjugate roots
    elif return_conjugate_realpart:
      roots.append(-0.25*a3)
      roots.append(-0.25*a3)
      roots.append(-0.25*a3)
      roots.append(-0.25*a3)
      return roots

    # D^2 >= 0: 2 real roots
    if square_D >= 0:
      D = math.sqrt(square_D)
      roots.append(-0.25*a3 + 0.5*(R + D))
      roots.append(-0.25*a3 + 0.5*(R - D))
    #  D^2 < 0: 2 complex conjugate roots
    elif return_conjugate_realpart:
      roots.append(-0.25*a3 + 0.5*R)
      roots.append(-0.25*a3 + 0.5*R)

    # E^2 >= 0: 2 real roots
    if square_E >= 0:
      E = math.sqrt(square_E)
      roots.append(-0.25*a3 - 0.5*(R - E))
      roots.append(-0.25*a3 - 0.5*(R + E))
    #  E^2 < 0: 2 complex conjugate roots
    elif return_conjugate_realpart:
      roots.append(-0.25*a3 - 0.5*R)
      roots.append(-0.25*a3 - 0.5*R)

    return roots

  # R < 0: 4 complex conjugate roots
  elif return_conjugate_realpart:
    R = math.sqrt(-square_R)

    # D^2 = A - iB
    # E^2 = A + iB
    A = 0.75*a3*a3 -square_R -2.0*a2
    B = 0.25*(4.0*a3*a2 -8.0*a1 - a3*a3*a3)/R

    # Compute the real parts of D and E (Real(D) = Real(E))
    # See: http://en.wikipedia.org/wiki/Complex_number#Square_root
    D = math.sqrt(0.5*(A + math.sqrt(A*A + B*B)))

    roots.append(-0.25*a3 - 0.5*D)
    roots.append(-0.25*a3 - 0.5*D)
    roots.append(-0.25*a3 + 0.5*D)
    roots.append(-0.25*a3 + 0.5*D)
    return roots

  return roots


def cos_poly(P):
  poly_size = len(P)+1

  temp = np.poly1d([0])
  coeffCos = np.poly1d([0])
  coeffSin = np.poly1d([0])
  tmpCoeffCos = np.poly1d([0])
  tmpCoeffSin = np.poly1d([0])

  cos0 = math.cos(P[0])
  sin0 = math.sin(P[0])

  coeffCos[0] = 1.0
  temp[0] = cos0

  # compute the derivative of P
  dP = np.polyder(P)

  facti = 1
  for i in xrange(1,poly_size):
    facti *= i

    tmpCoeffCos = np.polyder(coeffCos) + coeffSin * dP
    tmpCoeffSin = np.polyder(coeffSin) - coeffCos * dP

    coeffCos = tmpCoeffCos
    coeffSin = tmpCoeffSin

    temp[i] = (coeffCos[0] * cos0 + coeffSin[0] * sin0)/float(facti)

  return temp


def sin_poly(P):
  poly_size = len(P)+1

  temp = np.poly1d([0])
  coeffCos = np.poly1d([0])
  coeffSin = np.poly1d([0])
  tmpCoeffCos = np.poly1d([0])
  tmpCoeffSin = np.poly1d([0])

  cos0 = math.cos(P[0])
  sin0 = math.sin(P[0])

  coeffSin[0] = 1.0
  temp[0] = sin0

  # compute the derivative of P
  dP = np.polyder(P)

  facti = 1
  for i in xrange(1,poly_size):
    facti *= i

    tmpCoeffCos = np.polyder(coeffCos) + coeffSin * dP
    tmpCoeffSin = np.polyder(coeffSin) - coeffCos * dP

    coeffCos = tmpCoeffCos
    coeffSin = tmpCoeffSin

    temp[i] = (coeffCos[0] * cos0 + coeffSin[0] * sin0)/float(facti)

  return temp


###############################################################################


def test_root_computation():
  j = complex(0,1)
  print('############################ CUBIC ###################################')
  for compute_conjugate_realpart in (False,True):

    if compute_conjugate_realpart:
      print('###### Real roots + real part of complex conjugate roots ######')
    else:
      print('###### Real roots ######')
    print

    # From roots
    roots = np.array([[1, 1, 1], [1, 2, 3], [1, 3, 3],
                      [1, j, -j], [-5, 3*j, -3*j]]);
    for r in roots:
      sorted_r = np.sort(r)
      if not compute_conjugate_realpart:
        filtered_r = [_.real for _ in sorted_r if (not isinstance(_, complex))
                                        or (np.abs(_.imag) < 1e-12)]
      else:
        filtered_r = [_.real for _ in sorted_r]

      test_poly = np.poly(sorted_r)
      np_roots = np.roots(test_poly)
      if compute_conjugate_realpart:
        np_roots = [_.real for _ in np_roots]
      else:
        np_roots = [_.real for _ in np_roots if (not isinstance(_, complex))
                                             or (np.abs(_.imag) < 1e-6)]

      np_roots_sorted = np.sort(np_roots)
      print 'Actual roots: ', sorted_r
      print 'Numpy roots: ', np_roots_sorted
      analytical_roots = cubic_roots(test_poly, compute_conjugate_realpart)
      analytical_roots_sorted = np.sort(analytical_roots)
      print 'Analytical roots: ', analytical_roots_sorted
      assert(np.allclose(filtered_r, analytical_roots_sorted))
      print

    # from coeffs
    coeffs = np.array([[1, 1, 1, 1], [1, 2, 3, 4], [1, 3, 3, 4], [1, 0, 0, 0],
                       [1, 0, 0, 1], [1, -6, 11, -6], [1, 7, 49, 343],
                       [1.0, -3, -12.0, 24.0], [1.0, -11.0, 32.0, -28.0],
                       [1.0, 5.0, -16.0, -80.0], [1, 0, -3, 2]]);
    for c in coeffs:
      test_poly = np.poly1d(c)
      print test_poly
      np_roots = np.roots(test_poly.coeffs)
      if compute_conjugate_realpart:
        np_roots = [_.real for _ in np_roots]
      else:
        np_roots = [_.real for _ in np_roots if (not isinstance(_, complex))
                                             or (np.abs(_.imag) < 1e-6)]
      np_roots_sorted = np.sort(np_roots)
      print 'Numpy roots: ', np_roots_sorted
      analytical_roots = cubic_roots(test_poly.coeffs, compute_conjugate_realpart)
      analytical_roots_sorted = np.sort(analytical_roots)
      print 'Analytical roots: ', analytical_roots_sorted
      assert(np.allclose(np_roots_sorted, analytical_roots_sorted,
                         rtol=1e-05, atol=1e-08))
      print

  print('########################### QUARTIC ##################################')
  for compute_conjugate_realpart in (False,True):

    if compute_conjugate_realpart:
      print('###### Real roots + real part of complex conjugate roots ######')
    else:
      print('###### Real roots ######')
    print

    # From roots
    roots = np.array([[-2,-1,1,2], [1, 1, 1, 1], [1, 2, 3, 4], [1, 3, 3, 2],
                      [1, j, -j, -1], [-5*j, 5*j, -3*j, 3*j]]);
    for r in roots:
      sorted_r = np.sort(r)
      if not compute_conjugate_realpart:
        filtered_r = [_.real for _ in sorted_r if (not isinstance(_, complex))
                                        or (np.abs(_.imag) < 1e-12)]
      else:
        filtered_r = [_.real for _ in sorted_r]

      #filtered_r = np.sort(filtered_r)
      test_poly = np.poly(sorted_r)
      test_poly = test_poly.real
      print test_poly
      np_roots = np.roots(test_poly)
      if compute_conjugate_realpart:
        np_roots = [_.real for _ in np_roots]
      else:
        np_roots = [_.real for _ in np_roots if (not isinstance(_, complex))
                                             or (np.abs(_.imag) < 1e-6)]
      np_roots_sorted = np.sort(np_roots)
      print 'Actual roots: ', filtered_r
      print 'Numpy roots: ', np_roots_sorted
      analytical_roots = quartic_roots(test_poly, compute_conjugate_realpart)
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
      if compute_conjugate_realpart:
        np_roots = [_.real for _ in np_roots]
      else:
        np_roots = [_.real for _ in np_roots if (not isinstance(_, complex))
                                             or (np.abs(_.imag) < 1e-6)]
      np_roots_sorted = np.sort(np_roots)
      print 'Numpy roots: ', np_roots_sorted
      analytical_roots = quartic_roots(test_poly.coeffs, compute_conjugate_realpart)
      analytical_roots_sorted = np.sort(analytical_roots)
      print 'Analytical roots: ', analytical_roots_sorted
      assert(np.allclose(np_roots_sorted, analytical_roots_sorted,
                         rtol=1e-05, atol=1e-08))
      print

def test_cos_sin_polynomial():
  # Tested polynomials
  test_coeffs = [ [1,2,3,4,5], [1,1,1,-1,2]]
  for c in test_coeffs:
    P = np.poly1d(c[::-1])

    x = sp.symbols('x')
    symP = 0
    for k in xrange(len(c)):
      symP += P[k] * x**k

    ### FIRST: COSINE
    # Compute cos(P)
    res = cos_poly(P).c[::-1]

    # Compute cos(P) with symbolic expressions
    exp = sp.cos(symP)
    cosp = sp.series(exp, x, n=len(c))
    resSym = sp.collect(cosp.evalf(), x, evaluate=False, exact=False)

    # Compare the 2 results
    for i in xrange(len(c)):
      a = res[i]
      if i == 0:
        # big-O notation prevents comparison for the following test, so we
        # discard it
        b = resSym[x**0].subs(x,0).evalf()
      else:
        b = resSym[x**i]
      rel_tol = 1e-1
      abs_tol = 1e-1
      assert(abs(a-b) <= rel_tol*(abs(a)+abs(b)) + abs_tol)

    ### SECOND: SINE
    # Compute sin(P)
    res = sin_poly(P).c[::-1]

    # Compute sin(P) with symbolic expressions
    exp = sp.sin(symP)
    sinp = sp.series(exp, x, n=len(c))
    resSym = sp.collect(sinp.evalf(), x, evaluate=False, exact=False)

    # Compare the 2 results
    for i in xrange(len(c)):
      a = res[i]
      if i == 0:
        # big-O notation prevents comparison for the following test, so we
        # discard it
        b = resSym[x**0].subs(x,0).evalf()
      else:
        b = resSym[x**i]
      rel_tol = sys.float_info.epsilon
      abs_tol = sys.float_info.epsilon
      assert(abs(a-b) <= rel_tol*(abs(a)+abs(b)) + abs_tol)

if __name__ == "__main__":
  test_root_computation()
  test_cos_sin_polynomial()
