# -*- coding: utf-8 -*-
"""
@author: Benjamin Chrétien <chretien at lirmm dot fr>
"""

import numpy as np
import sympy as sp
import copy
import math
from scipy import misc as sc
from fractions import Fraction

def compute_open_uniform_bspline_matrix( degree, nctrl, infBound, supBound ):
  """
  Compute the coefficients of the Bézier basis functions
  We chose the following notation:
    B_n(t) = [P0 P1 ... Pn] Bspline_mat [1 t ... t^n]^T
  """

  # Variable initialization
  nbCtlPoints = max(degree + 1,nctrl)
  order = degree
  knotVectorSize = nbCtlPoints + order + 1

  prevSplines = np.zeros([knotVectorSize, degree + 1])
  splines = np.zeros([knotVectorSize, degree + 1])

  # knot-vector : open uniform
  # => the curve starts at the first control point and ends at the last
  #    control point
  knotvector = np.zeros(knotVectorSize)
  counter = 0

  # i <= order : t_{i} = t1
  for i in xrange(0,order):
      knotvector[counter] = infBound
      counter = counter + 1

  # order <= i <= nbCtrlPoints : t_{i+1} - t_{i} = constant
  for i in xrange(0,nbCtlPoints - order+1):
      knotvector[counter] = infBound + i *(supBound - infBound) / (nbCtlPoints - order)
      counter = counter + 1

  # i > nbCtrlPoints : t_{i} = t_{order+nbCtrlPoints}
  for i in xrange(0,order):
      knotvector[counter] = supBound
      counter = counter + 1

  # set prevSplines ---> B_{j,0}
  for j in xrange(0,knotVectorSize-1):
     if knotvector[j] != knotvector[j+1]:
         prevSplines[j][0] = 1
     else:
         prevSplines[j][0] = 0

  B1_denominator = 0
  B2_denominator = 0

  # the two parts used to compute the basis functions
  B1 = np.zeros(degree + 1)
  B2 = np.zeros(degree + 1)

  # numerator : t - t_{j} or t_{j+k+1} - t
  B1_numerator = np.zeros(degree + 1)
  B1_resultingNumerator = np.zeros(degree + 1)
  B2_numerator = np.zeros(degree + 1)
  B2_resultingNumerator = np.zeros(degree + 1)

  # bottom-up approach ---> B_{j,k}
  for k in xrange(1,order+1):
       for j in xrange(0,knotVectorSize-k-1):
          # compute the first part of the basis function:
          # B1 = (t - t_{j})/(t_{j+k} - t_{j}) * B_{j,k-1}
          B1 = np.zeros(degree + 1)
          # t_{j+k} - t_{j}
          B1_denominator = (knotvector[j+k] - knotvector[j])
          if (B1_denominator * B1_denominator) > 1e-6:
            # t - t_{j}
            B1_numerator = np.zeros(degree + 1)
            B1_numerator[1] = 1
            B1_numerator[0] = -knotvector[j]

            B1_resultingNumerator = np.zeros(degree + 1)
            # resultingNumerator = numerator * B_{j,k-1}
            # /!\ python polynonial ordered from highest to lowest degree
            B1_tmp = np.polymul(B1_numerator[::-1], prevSplines[j][::-1])[::-1][0:degree+1]

            for i in xrange(len(B1_tmp)):
              B1_resultingNumerator[i] = B1_tmp[i]

            B1 = B1_resultingNumerator / B1_denominator

          # compute the second part of the basis function:
          # B2 = (t_{j+k+1} - t)/(t_{j+k+1} - t_{j+1}) * B_{j+1,k-1}
          B2 = np.zeros(degree + 1)
          # (t_{j+k+1} - t_{j+1})
          B2_denominator = (knotvector[j+k+1] - knotvector[j+1])
          if (B2_denominator * B2_denominator) > 1e-6:
            # t_{j+k+1} - t
            B2_numerator = np.zeros(degree + 1)
            B2_numerator[1] = -1
            B2_numerator[0] = knotvector[j+k+1]

            # resultingNumerator = numerator * B_{j+1,k-1}
            B2_resultingNumerator = np.zeros(degree + 1)
            # /!\ python polynonial ordered from highest to lowest degree
            B2_tmp = np.polymul(B2_numerator[::-1], prevSplines[j+1][::-1])[::-1][0:degree+1]

            for i in xrange(len(B2_tmp)):
              B2_resultingNumerator[i] = B2_tmp[i]

            B2 = B2_resultingNumerator / B2_denominator

          splines[j][:] = B1 + B2

       # required for the recursion
       prevSplines = copy.copy(splines)

  return splines[0:nbCtlPoints][:]

def compute_uniform_bspline_matrix( k ):
  """
  Based on "General matrix representations for B-splines" by Kaihuai Qin
  Notation used: [1 u u^2 ... u^(k-1)] M [P0 P1 ... Pn]^T
  """
  # degree = k-1
  M = np.zeros([k , k])
  for i in xrange(k):
    for j in xrange(k):
      for s in xrange(j,k):
        # (-1)^(s-j) * C(s-j, k) * (k-s-1)^(k-1-i)
        M[i][j] += math.pow(-1, s-j) * sc.comb(k, s-j, exact=True) * math.pow(k-s-1, k-1-i)
      # C(k-1-i, k-1)
      M[i][j] *= sc.comb(k-1, k-1-i, exact=True)

  # 1/(k-1)!
  M /= math.factorial(k-1)
  return M


def cubic_span_matrix():
  """
  Based on "A practical review of uniform b-splines" by Kristin Branson
  Notation used: [1 s s^2 s^3] B [P0 P1 P2 P3]^T
  See: http://vision.ucsd.edu/~kbranson/research/bsplines.html
  """

  # Symbolic variables
  i = sp.Symbol('i')
  s = sp.Symbol('s')
  P = sp.MatrixSymbol('p', 4, 1)

  # Use the result given in the paper
  # TODO: implement that part that leads to this result
  res = Fraction(1,6)*( (1-(s-i))**3 * P[0,0]
            + (3*(s-i)**3 - 6*(s-i)**2 + 4) * P[1,0]
            + (-3*(s-i)**3 + 3*(s-i)**2 + 3*(s-i) + 1) * P[2,0]
            + (s-i)**3 * P[3,0]
            )
  res = res.expand()
  # Handle as a symbolic polynomial
  res = sp.Poly(res, P[0,0], P[1,0], P[2,0], P[3,0])

  # Span matrix Bi
  Bi = sp.Matrix(4,4, lambda i,j: 0)

  # For each column (control points Pj)
  for col in xrange(4):
    basis = sp.zeros(4)
    basis[col] = 1
    # Substitute by the right vector to get the proper coefficient
    # (there may be a better way to achieve that with sympy)
    p_ctrl_pt = res.subs( {P[0,0]: basis[0],
                           P[1,0]: basis[1],
                           P[2,0]: basis[2],
                           P[3,0]: basis[3]})
    coeffs = sp.Poly(p_ctrl_pt, s).as_poly().all_coeffs()[::-1]

    # For each row (s^k)
    for row in xrange(len(coeffs)):
      # Store the factorized result in Bi
      Bi[row,col] = coeffs[row].factor(i)

  return Bi

def main ():
  # Print with the [1 t t^2 ...] M [P0 P1 P2 ...]^T notation
  print compute_uniform_bspline_matrix(3)
  print
  print compute_uniform_bspline_matrix(4)
  print
  sp.pretty_print(cubic_span_matrix())

if __name__ == "__main__":
  main ()
