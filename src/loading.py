#!/usr/bin/env python

"""
    Filename: loading.py
    Author: Alex Shokolarov
    Date last modified: 26/02/2021
    Python version: 3.8
"""

import numpy as np
from params import *
import pandas as pd
from scipy import interpolate
from scipy import integrate
from bending_stiffness import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tools import *
import matplotlib.path as mpath



def func0(x, x1):
    #Macaulay step function of order 0.
    if x >= x1:
        return 1
    else:
        return 0

def func1(x, x1):
    # Macaulay step function of order 1.
    if x >= x1:
        return x - x1
    else:
        return 0


class Loading:

    def __init__(self, ail, datafile, **load_params):
        """
        Loading class used to compute internal loading, deflections and stresses in the aileron.
        :param ail: aileron class representing the geometry of the aileron
        :param datafile: datafile containing the aerodynamic loading data
        :param load_params: loading parameters and boundary conditions
        """
        self.ail = ail
        self.datafile = datafile
        self.P = load_params["P"]
        self.theta = load_params["theta"]
        self.d1 = load_params["d1"]
        self.d3 = load_params["d3"]
        self.Nz = load_params["Nz"]
        self.Nx = load_params["Nx"]
        self.w = self._compute_uniform()[0]
        self.Izz, self.Iyy = self.ail.SMOA
        self.zbar = self.ail.shear_center
        self.J = self.ail.torsional_stiffness

    def _interploate_loading(self):
        """
        Compute continuous function of the aerodynamic load
        by performing a bivariate interpolation using radial basis functions method.

        :return: f(z, x) to compute load F at point [z,x]
        """
        # Allocate arrays for coordinates
        Nz = self.Nz
        Nx = self.Nx

        z = np.zeros(Nz)
        x = np.zeros(Nx)

        # Compute z coordinates
        for i in range(1, Nz):
            t1 = (i - 1) * np.pi / Nz
            t2 = i * np.pi / Nz
            z[i] = -0.5 * (self.ail.Ca / 2 * (1 - np.cos(t1)) + self.ail.Ca / 2 * (1 - np.cos(t2)))
        z = -1 + 2 * (z - np.min(z)) / (np.max(z) - np.min(z))

        # Compute z coordinates
        for i in range(1, Nx):
            t1 = (i - 1) * np.pi / Nx
            t2 = i * np.pi / Nx
            x[i] = -0.5 * (self.ail.la / 2 * (1 - np.cos(t1)) + self.ail.la / 2 * (1 - np.cos(t2)))
        x = -1 + 2 * (x - np.min(x)) / (np.max(x) - np.min(x))

        # Create mesh and read data
        X, Z = np.meshgrid(x, z)
        Y = pd.read_csv(self.datafile, header=None)

        # https: // docs.scipy.org / doc / scipy / reference / generated / scipy.interpolate.Rbf.html
        f = interpolate.Rbf(Z, X, Y, function='cubic', smooth=0)
        return f

    def _compute_total_aero_load(self):
        """
        Integrate the interpolated aerodynamic load to find the total aero load
        and compute an estimate for the error in it.
        :return: L[0] - total load [kN], L[1] - estimate for the error in the computed load
        """
        f = self._interploate_loading()
        L = integrate.dblquad(f, -1, 1, lambda x: -1, lambda x: 1, epsabs=0.1)
        return np.array(L)

    def _compute_uniform(self):
        """
        Compute uniform load over hinge line by diving total load over the span.
        :return: uniform[0] - uniform loading over hinge line [N/m], uniform[1] - estimate of error in the computation
        """
        total = self._compute_total_aero_load()
        uniform = (total / self.ail.la) * 1E3
        return uniform

    def My(self, x):
        """
        Obtain row and constant term relating the reaction forces to the moment in Y.
        :param x: position along the span
        """
        _A = np.zeros(12)
        _A[0] = func1(x, self.ail.x1)
        _A[1] = -np.cos(self.theta) * func1(x, self.ail.x2 - self.ail.xa/2)
        _A[2] = func1(x, self.ail.x2)
        _A[3] = func1(x, self.ail.x3)

        _b = -self.P * np.cos(self.theta) * func1(x, self.ail.x2 + self.ail.xa/2)
        return _A, _b

    def Mz(self, x):
        """
        Obtain row and constant term relating the reaction forces to the moment in Z.
        :param x: position along the span
        """
        _A = np.zeros(12)
        _A[1] = -np.sin(self.theta) * func1(x, self.ail.x2 - self.ail.xa/2)
        _A[4] = func1(x, self.ail.x1)
        _A[5] = func1(x, self.ail.x2)
        _A[6] = func1(x, self.ail.x3)

        _b = -self.P * np.sin(self.theta) * func1(x, self.ail.x2 + self.ail.xa/2) -\
            self.w * x * x / 2

        return _A, _b

    def T(self, x):
        """
        Obtain row and constant term relating the reaction forces to the torque about X.
        :param x: position along the span
        """
        zbar = self.ail.shear_center
        _A = np.zeros(12)
        _A[1] = -np.sin(self.theta) * zbar * func0(x, self.ail.x2 - self.ail.xa/2)
        _A[4] = func0(x, self.ail.x1) * (zbar - self.ail.ha/2)
        _A[5] = func0(x, self.ail.x2) * (zbar - self.ail.ha/2)
        _A[6] = func0(x, self.ail.x3) * (zbar - self.ail.ha/2)

        _b = -self.P*np.sin(self.theta)*zbar*func0(x, self.ail.x2 + self.ail.xa/2) - self.w * (zbar - self.ail.ha/2) * x
        return _A, _b

    def twist(self, x):
        """
        Obtain row and constant term relating the reaction forces to the twist about X.
        :param x: position along the span
        """
        zbar = self.ail.shear_center
        _A = np.zeros(12)
        _A[1] = -np.sin(self.theta) * zbar * func1(x, self.ail.x2 - self.ail.xa/2)
        _A[4] = func1(x, self.ail.x1) * (zbar - self.ail.ha / 2)
        _A[5] = func1(x, self.ail.x2) * (zbar - self.ail.ha / 2)
        _A[6] = func1(x, self.ail.x3) * (zbar - self.ail.ha / 2)
        _A *= 1 / self.ail.G / self.J
        _A[11] = 1

        _b = -self.P*np.sin(self.theta)*zbar*func1(x, self.ail.x2 + self.ail.xa/2) - self.w * (zbar - self.ail.ha/2) * x * x / 2
        _b *= 1 / self.ail.G / self.J
        return _A, _b

    def Sz(self, x):
        """
        Obtain row and constant term relating the reaction forces to the force in Z
        :param x: position along the span
        """
        _A = np.zeros(12)
        _A[0] = func0(x, self.ail.x1)
        _A[1] = -np.cos(self.theta) * func0(x, self.ail.x2 - self.ail.xa/2)
        _A[2] = func0(x, self.ail.x2)
        _A[3] = func0(x, self.ail.x3)

        _b = -self.P*np.cos(self.theta)*func0(x, self.ail.x2 + self.ail.xa/2)
        return _A, _b

    def Sy(self, x):
        """
        Obtain row and constant term relating the reaction forces to the force in y
        :param x: position along the span
        """
        _A = np.zeros(12)
        _A[1] = -np.sin(self.theta) * func0(x, self.ail.x2 - self.ail.xa/2)
        _A[4] = func0(x, self.ail.x1)
        _A[5] = func0(x, self.ail.x2)
        _A[6] = func0(x, self.ail.x3)

        _b = -self.P*np.sin(self.theta)*func0(x, self.ail.x2 + self.ail.xa/2) - self.w * x
        return _A, _b

    def dy(self, x):
        """
        Obtain row and constant term relating the reaction forces to the deflection in y.
        :param x: position along the span
        """
        _A = np.zeros(12)
        _A[1] = -np.sin(self.theta)/6 * func1(x, self.ail.x2 - self.ail.xa/2) ** 3
        _A[4] = (1/6)*func1(x, self.ail.x1) ** 3
        _A[5] = (1/6)*func1(x, self.ail.x2) ** 3
        _A[6] = (1/6)*func1(x, self.ail.x3) ** 3
        _A *= -1 / self.ail.E / self.Izz
        _A[7] = x
        _A[8] = 1

        _b = -self.P/6 * np.sin(self.theta) * func1(x, self.ail.x2 + self.ail.xa/2) ** 3 - self.w * (x ** 4) / 24
        _b *= -1 / self.ail.E / self.Izz

        return _A, _b

    def dydx(self, x):
        """
        Obtain row and constant term relating the reaction forces to the slope in y.
        :param x: position along the span
        """
        _A = np.zeros(12)
        _A[1] = -(1/2)*np.sin(self.theta) * func1(x, self.ail.x2 - self.ail.xa/2) ** 2
        _A[4] = (1/2) * func1(x, self.ail.x1) ** 2
        _A[5] = (1/2) * func1(x, self.ail.x2) ** 2
        _A[6] = (1/2) * func1(x, self.ail.x3) ** 2
        _A *= -1 / self.ail.E / self.Izz
        _A[7] = 1

        _b = -self.P/2 * np.sin(self.theta) * func1(x, self.ail.x2 + self.ail.xa/2) ** 2 - self.w * (x ** 3) / 6
        _b *= -1 / self.ail.E / self.Izz

        return _A, _b

    def dz(self, x):
        """
        Obtain row and constant term relating the reaction forces to the deflection in z.
        :param x: position along the span
        """
        _A = np.zeros(12)
        _A[0] = (1/6) * func1(x, self.ail.x1) ** 3
        _A[1] = -(1/6) * np.cos(self.theta) * func1(x, self.ail.x2 - self.ail.xa/2) ** 3
        _A[2] = func1(x, self.ail.x2) ** 3 / 6
        _A[3] = func1(x, self.ail.x3) ** 3 / 6
        _A *= -1 / self.ail.E / self.Iyy
        _A[9] = x
        _A[10] = 1

        _b = -(1/6) * self.P * np.cos(self.theta) * func1(x, self.ail.x2 + self.ail.xa/2) ** 3
        _b *= -1 / self.ail.E / self.Iyy

        return _A, _b

    def dzdx(self, x):
        """
        Obtain row and constant term relating the reaction forces to the slope in z.
        :param x: position along the span
        """
        _A = np.zeros(12)
        _A[0] = (1/2)*func1(x, self.ail.x1) ** 2
        _A[1] = -(1/2)*np.cos(self.theta) * func1(x, self.ail.x2 - self.ail.xa / 2) ** 2
        _A[2] = (1/2)*func1(x, self.ail.x2) ** 2
        _A[3] = (1/2)*func1(x, self.ail.x3) ** 2
        _A *= -1 / self.ail.E / self.Iyy
        _A[9] = 1

        _b = -(1/2) * self.P * np.cos(self.theta) * func1(x, self.ail.x2 + self.ail.xa/2) ** 2
        _b *= -1 / self.ail.E / self.Iyy

        return _A, _b

    def solve_equilibrium(self):
        """
        Solve the equilibrium problem and obtain the reaction forces and integration coefficients.
        Solution vector of the form X = <Rz1, A, Rz2, Rz3, Ry1, Ry2, Ry3, C1, C2, C3, C4, C5>.
        """
        A = np.zeros((12,12))
        b = np.zeros(12)

        A1, b1 = self.My(self.ail.la)
        A2, b2 = self.Mz(self.ail.la)
        A3, b3 = self.T(self.ail.la)
        A4, b4 = self.Sz(self.ail.la)
        A5, b5 = self.Sy(self.ail.la)
        A6, b6 = self.dy(self.ail.x1)
        A7, b7 = self.dz(self.ail.x1)
        A8, b8 = self.dy(self.ail.x2)
        A9, b9 = self.dz(self.ail.x2)
        A10, b10 = self.dy(self.ail.x3)
        A11, b11 = self.dz(self.ail.x3)
        A12, b12 = self.twist(self.ail.x2 - self.ail.xa/2)

        A[0, :] = A1
        A[1, :] = A2
        A[2, :] = A3
        A[3, :] = A4
        A[4, :] = A5
        A[5, :] = A6
        A[6, :] = A7
        A[7, :] = A8
        A[8, :] = A9
        A[9, :] = A10
        A[10, :] = A11
        A[11, :] = A12

        b[0] = -b1
        b[1] = -b2
        b[2] = -b3
        b[3] = -b4
        b[4] = -b5
        b[5] = -self.d1 * np.cos(self.theta) - b6
        b[6] = -self.d1 * np.sin(self.theta) - b7
        b[7] = -b8
        b[8] = -b9
        b[9] = -self.d3 * np.cos(self.theta) - b10
        b[10] = -self.d3 * np.sin(self.theta) - b11
        b[11] = -b12

        X = np.linalg.solve(A, b)
        return X

    def qby(self, Sy):
        """
        Compute the shear flow distribution in the cross-section due to a shear force in the y direction.
        :param Sy: Shear force in y
        """
        Izz = self.Izz

        r = self.ail.ha / 2
        sl = self.ail.sl

        A = sum([stiff.A for stiff in self.ail.stiffeners])
        P = np.pi * r + 2 * sl
        t_add = A / P
        tsk = self.ail.tsk + t_add
        tsp = self.ail.tsp

        qb1 = lambda x: (Sy * tsk * r * r / Izz) * (np.cos(x) - 1)
        qb2 = lambda x: -(Sy * tsp / 2 / Izz) * x * x
        qb3 = lambda x: -(Sy * tsk / Izz) * (r * x - (r / 2 / sl) * x * x) + qb1(np.pi / 2) + qb2(r)
        qb4 = lambda x: (Sy * tsk / 2 / Izz) * (r * x * x / sl) + qb3(sl)
        qb5 = lambda x: -(Sy * tsp / 2 / Izz) * x * x
        qb6 = lambda x: (Sy * tsk * r * r / Izz) * np.cos(x) + qb4(sl) - qb5(-r)

        A = np.zeros((2, 2))
        A[0, 0] = r * np.pi / tsk + 2 * r / tsp
        A[0, 1] = -2 * r / tsp
        A[1, 0] = -2 * r / tsp
        A[1, 1] = 2 * sl / tsk + 2 * r / tsp

        b = np.zeros(2)
        qb1_int = lambda x: qb1(x) * r / tsk
        qb2_int = lambda x: qb2(x) / tsp
        qb6_int = lambda x: (Sy * tsk * r * r / Izz) * np.cos(x) + (qb4(sl) - qb5(-r))*(1+np.pi/2)*r
        b[0] = quad(qb1_int, 0, np.pi / 2)[0] - quad(qb2_int, 0, r)[0] - quad(qb2_int, 0, r)[0] + \
               quad(qb6_int, -np.pi / 2, 0)[0]
        qb3_int = lambda x: qb3(x) / tsk
        qb4_int = lambda x: qb4(x) / tsk
        qb5_int = lambda x: qb5(x) / tsp
        b[1] = quad(qb2_int, 0, r)[0] + quad(qb3_int, 0, sl)[0] + quad(qb4_int, 0, sl)[0] + \
               quad(qb5_int, 0, r)[0]

        qr0, qr1 = np.linalg.solve(A, -b)

        q1 = lambda x: qb1(x) + qr0
        q2 = lambda x: qb2(x) - qr0 + qr1
        q3 = lambda x: qb3(x) + qr1
        q4 = lambda x: qb4(x) + qr1
        q5 = lambda x: qb5(x) - qr0 + qr1
        q6 = lambda x: qb6(x) + qr0

        return [q1,q2,q3,q4,q5,q6]

    def qbz(self, Sz):
        """
        Compute the shear flow distribution in the cross-section due to a shear force in the z direction.
        :param Sz: Shear force in z
        """
        Iyy = self.Iyy
        z = self.ail.centroid[0]

        r = self.ail.ha / 2
        sl = self.ail.sl

        A = sum([stiff.A for stiff in self.ail.stiffeners])
        P = np.pi * r + 2 * sl
        t_add = A / P
        tsk = self.ail.tsk + t_add
        tsp = self.ail.tsp

        qb1 = lambda x: (Sz*tsk/Iyy) * (-r*r*np.sin(x) + r*r*x - z*r*x)
        qb2 = lambda x: -(Sz*tsp/Iyy) * (z - r) * x
        qb3 = lambda x: -(Sz*tsk/Iyy) * (-(r-z)*x - (self.ail.Ca - r)/(2*sl) * x * x) + qb1(np.pi/2) + qb2(r)
        qb4 = lambda x: (Sz*tsk/Iyy) * ((self.ail.Ca-z)*x - (self.ail.Ca - r)/(2*sl) * x * x) + qb3(sl)
        qb5 = lambda x: (Sz * tsp / Iyy) * (r - z) * x
        qb6 = lambda x: -(Sz*tsk/Iyy) * (r*r*(np.sin(x)+1) - r*r*(x+np.pi/2) + z*r*(x+np.pi/2)) + qb4(sl) - qb5(-r)

        return [qb1,qb2,qb3,qb4,qb5,qb6]

    def qbT(self, T):
        """
        Compute the shear flow distribution in the cross-section due to a torque about the x-axis
        :param T: Torque about x
        """
        r = self.ail.ha / 2
        A1 = np.pi * r ** 2 / 2
        A2 = (self.ail.Ca - r) * r

        A = sum([stiff.A for stiff in self.ail.stiffeners])
        P = np.pi * r + 2 * self.ail.sl
        t_add = A / P
        tsk = self.ail.tsk + t_add
        tsp = self.ail.tsp

        A = np.zeros((3, 3))
        A[0, 0] = 2 * A1
        A[0, 1] = 2 * A2
        A[1, 0] = (r * np.pi / tsk + 2 * r / tsp) / (2 * A1)
        A[1, 1] = (-2 * r / tsp) / (2 * A1)
        A[1, 2] = -1
        A[2, 0] = (-2 * r / tsp) / (2 * A2)
        A[2, 1] = (2 * self.ail.sl / tsk + 2 * r / tsp) / (2 * A2)
        A[2, 2] = -1

        b = np.zeros(3)
        b[0] = 1
        b[1] = 0
        b[2] = 0

        x = np.linalg.solve(A, b)
        return x[:2] * T

    def total_shear(self, Sy, Sz, T):
        """
        Combine the shear distribution from Sy, Sz, T and produce a final shear flow distribution.
        :param Sy: Shear force in y.
        :param Sz: Shear force in z.
        :param T: Torque about x.
        """
        qT1, qT2 = self.qbT(T)

        qy = self.qby(Sy)
        qz = self.qbz(Sz)

        q1 = lambda x: qy[0](x) + qz[0](x) - qT1
        q2 = lambda x: qy[1](x) + qz[1](x) + qT1 - qT2
        q3 = lambda x: qy[2](x) + qz[2](x) - qT2
        q4 = lambda x: qy[3](x) + qz[3](x) - qT2
        q5 = lambda x: qy[4](x) + qz[4](x) + qT1 - qT2
        q6 = lambda x: qy[5](x) + qz[5](x) - qT1

        return q1,q2,q3,q4,q5,q6

    def compute_stresses(self, x=None, loading=None):
        """
        Compute the direct, shear and von mises stresses in the cross-section at a certain location/loading.
        :param x: location along the span
        :param loading: loading of the form [My,Mz,Sy,Sz,T]
        """
        N = 50
        r = self.ail.ha/2
        sl = self.ail.sl
        zbar = self.ail.centroid[0]

        sec1 = np.linspace(0, np.pi / 2, N)
        sec2 = np.linspace(0, r, N)
        sec3 = np.linspace(0, sl, N)
        sec4 = np.linspace(0, sl, N)
        sec5 = np.linspace(0, -r, N)
        sec6 = np.linspace(-np.pi / 2, 0, N)

        z1, y1 = r + -r*np.cos(sec1), r*np.sin(sec1)
        z2, y2 = np.ones(N) * r, sec2
        z3, y3 = r + np.cos(self.ail.psi) * sec3, r - np.sin(self.ail.psi) * sec3
        z4, y4 = self.ail.Ca - np.cos(self.ail.psi) * sec4, -np.sin(self.ail.psi) * sec4
        z5, y5 = np.ones(N) * r, sec5
        z6, y6 = r - r*np.cos(sec6), r*np.sin(sec6)

        if loading is None:
            X = self.solve_equilibrium()

            # Direct stress calculations
            _A, _b = self.My(x)
            My = np.dot(_A, X) + _b

            _A, _b = self.Mz(x)
            Mz = np.dot(_A, X) + _b
            Mz = -Mz

            # Shear stresses
            _A, _b = self.Sy(x)
            Sy = np.dot(_A, X) + _b
            Sy = -Sy

            _A, _b = self.Sz(x)
            Sz = np.dot(_A, X) + _b
            Sz = -Sz

            _A, _b = self.T(x)
            T = np.dot(_A, X) + _b
            T = -T

        else:
            My,Mz,Sy,Sz,T = loading

        zbar = self.ail.centroid[0]

        sig = np.zeros((6, N))
        for j in range(N):
            sig[0][j] = -My * (z1[j] - zbar) / self.Iyy + Mz * (y1[j]) / self.Izz
            sig[1][j] = -My * (z2[j] - zbar) / self.Iyy + Mz * (y2[j]) / self.Izz
            sig[2][j] = -My * (z3[j] - zbar) / self.Iyy + Mz * (y3[j]) / self.Izz
            sig[3][j] = -My * (z4[j] - zbar) / self.Iyy + Mz * (y4[j]) / self.Izz
            sig[4][j] = -My * (z5[j] - zbar) / self.Iyy + Mz * (y5[j]) / self.Izz
            sig[5][j] = -My * (z6[j] - zbar) / self.Iyy + Mz * (y6[j]) / self.Izz

        q1, q2, q3, q4, q5, q6 = self.total_shear(Sy,Sz,T)

        ss = np.zeros((6, N))

        for j in range(N):
            ss[0][j] = q1(sec1[j])
            ss[1][j] = q2(sec2[j])
            ss[2][j] = q3(sec3[j])
            ss[3][j] = q4(sec4[j])
            ss[4][j] = q5(sec5[j])
            ss[5][j] = q6(sec6[j])

        vm = np.zeros((6, N))
        for j in range(N):
            vm[0][j] = np.sqrt(sig[0][j] ** 2 + 3 * (ss[0][j] / self.ail.tsk)**2)
            vm[1][j] = np.sqrt(sig[1][j] ** 2 + 3 * (ss[1][j] / self.ail.tsp)**2)
            vm[2][j] = np.sqrt(sig[2][j] ** 2 + 3 * (ss[2][j] / self.ail.tsk)**2)
            vm[3][j] = np.sqrt(sig[3][j] ** 2 + 3 * (ss[3][j] / self.ail.tsk)**2)
            vm[4][j] = np.sqrt(sig[4][j] ** 2 + 3 * (ss[4][j] / self.ail.tsp)**2)
            vm[5][j] = np.sqrt(sig[5][j] ** 2 + 3 * (ss[5][j] / self.ail.tsk)**2)

        return sig, ss, vm

    def plot_crossection(self, x=None, loading=None):
        """
        Plot the direct, shear and von mises stresses in the cross-section at a certain location/loading.
        :param x: location along the span.
        :param loading: loading of the form [My,Mz,Sy,Sz,T]
        """
        N = 50
        r = self.ail.ha / 2
        sl = self.ail.sl
        zbar = self.ail.centroid[0]

        sec1 = np.linspace(0, np.pi / 2, N)
        sec2 = np.linspace(0, r, N)
        sec3 = np.linspace(0, sl, N)
        sec4 = np.linspace(0, sl, N)
        sec5 = np.linspace(0, -r, N)
        sec6 = np.linspace(-np.pi / 2, 0, N)

        z1, y1 = r + -r * np.cos(sec1), r * np.sin(sec1)
        z2, y2 = np.ones(N) * r, sec2
        z3, y3 = r + np.cos(self.ail.psi) * sec3, r - np.sin(self.ail.psi) * sec3
        z4, y4 = self.ail.Ca - np.cos(self.ail.psi) * sec4, -np.sin(self.ail.psi) * sec4
        z5, y5 = np.ones(N) * r, sec5
        z6, y6 = r - r * np.cos(sec6), r * np.sin(sec6)

        if loading is None:
            sig, ss, vm = self.compute_stresses(x=x)
        else:
            sig, ss, vm = self.compute_stresses(loading=loading)

        #Shear stress
        ### Plot region 1
        path = mpath.Path(np.column_stack([z1, y1]))
        verts = path.interpolated(steps=1).vertices
        z1, y1 = verts[:, 0, ], verts[:, 1]
        maxabs = np.max(np.abs(ss[0]))

        ### Plot region 2
        path = mpath.Path(np.column_stack([z2, y2]))
        verts = path.interpolated(steps=1).vertices
        z2, y2 = verts[:, 0, ], verts[:, 1]
        maxabs2 = np.max(np.abs(ss[1]))
        maxabs = max(maxabs2, maxabs)

        ### Plot region 3
        path = mpath.Path(np.column_stack([z3, y3]))
        verts = path.interpolated(steps=1).vertices
        z3, y3 = verts[:, 0, ], verts[:, 1]
        maxabs3 = np.max(np.abs(ss[2]))
        maxabs = max(maxabs3, maxabs)

        ### Plot region 4
        path = mpath.Path(np.column_stack([z4, y4]))
        verts = path.interpolated(steps=1).vertices
        z4, y4 = verts[:, 0, ], verts[:, 1]
        maxabs4 = np.max(np.abs(ss[3]))
        maxabs = max(maxabs4, maxabs)

        ### Plot region 5
        path = mpath.Path(np.column_stack([z5, y5]))
        verts = path.interpolated(steps=1).vertices
        z5, y5 = verts[:, 0, ], verts[:, 1]
        maxabs5 = np.max(np.abs(ss[4]))
        maxabs = max(maxabs5, maxabs)

        ### Plot region 6
        path = mpath.Path(np.column_stack([z6, y6]))
        verts = path.interpolated(steps=1).vertices
        z6, y6 = verts[:, 0, ], verts[:, 1]
        maxabs6 = np.max(np.abs(ss[5]))
        maxabs = max(maxabs6, maxabs)
        fig = plt.figure(4)

        colorline(z1, y1, ss[0], cmap=plt.get_cmap('jet'),
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        colorline(z2, y2, ss[1], cmap=plt.get_cmap('jet'),
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        colorline(z3, y3, ss[2], cmap=plt.get_cmap('jet'),
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        colorline(z4, y4, ss[3], cmap=plt.get_cmap('jet'),
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        colorline(z5, y5, ss[4], cmap=plt.get_cmap('jet'),
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        colorline(z6, y6, ss[5], cmap=plt.get_cmap('jet'),
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)

        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('jet'),
                                   norm=plt.Normalize(-maxabs, maxabs))
        sm.set_array([])
        plt.colorbar(sm, label=r'$q$ [N/m]', fraction=0.20, pad=0.04,
                     orientation="horizontal")
        plt.xlim(-self.ail.Ca - 0.1, 0.1)
        plt.ylim(-self.ail.ha / 2 - 0.02, self.ail.ha / 2 + 0.02)
        plt.axis('scaled')
        plt.xlabel(r'$-z$ [m]')
        plt.ylabel(r'$y$ [m]')
        if x is not None:
            plt.title(f'Numerical Shear flow distribution at {x}m span-wise')
        else:
            plt.title(f'Numerical Shear flow distribution')
        plt.show()

        # Direct stresses
        maxabs = np.max(np.abs(sig[0]))

        ### Plot region 2
        maxabs2 = np.max(np.abs(sig[1]))
        maxabs = max(maxabs2, maxabs)

        ### Plot region 3
        maxabs3 = np.max(np.abs(sig[2]))
        maxabs = max(maxabs3, maxabs)

        ### Plot region 4
        maxabs4 = np.max(np.abs(sig[3]))
        maxabs = max(maxabs4, maxabs)

        ### Plot region 5
        maxabs5 = np.max(np.abs(sig[4]))
        maxabs = max(maxabs5, maxabs)

        ### Plot region 6
        maxabs6 = np.max(np.abs(sig[5]))
        maxabs = max(maxabs6, maxabs)

        print(f"Maximum Von Mises stress of {maxabs/1e6}MPa.")

        fig = plt.figure(4)

        colorline(z1, y1, sig[0], cmap=plt.get_cmap('jet'),
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        colorline(z2, y2, sig[1], cmap=plt.get_cmap('jet'),
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        colorline(z3, y3, sig[2], cmap=plt.get_cmap('jet'),
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        colorline(z4, y4, sig[3], cmap=plt.get_cmap('jet'),
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        colorline(z5, y5, sig[4], cmap=plt.get_cmap('jet'),
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)
        colorline(z6, y6, sig[5], cmap=plt.get_cmap('jet'),
                  norm=plt.Normalize(-maxabs, maxabs), linewidth=2)

        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('jet'),
                                   norm=plt.Normalize(-maxabs, maxabs))
        sm.set_array([])
        plt.colorbar(sm, label=r'$\sigma_{xx}$ N/m$^2$', fraction=0.20, pad=0.04,
                     orientation="horizontal")
        plt.xlim(-self.ail.Ca - 0.1, 0.1)
        plt.ylim(-self.ail.ha / 2 - 0.02, self.ail.ha / 2 + 0.02)
        plt.axis('scaled')
        plt.xlabel(r'$-z$ [m]')
        plt.ylabel(r'$y$ [m]')
        if x is not None:
            plt.title(f'Numerical Direct stress distribution at {x}m span-wise')
        else:
            plt.title(f'Numerical Direct stress distribution')
        plt.show()

        # Von Mises stresses
        maxabs = np.max(np.abs(vm[0]))

        ### Plot region 2
        maxabs2 = np.max(np.abs(vm[1]))
        maxabs = max(maxabs2, maxabs)

        ### Plot region 3
        maxabs3 = np.max(np.abs(vm[2]))
        maxabs = max(maxabs3, maxabs)

        ### Plot region 4
        maxabs4 = np.max(np.abs(vm[3]))
        maxabs = max(maxabs4, maxabs)

        ### Plot region 5
        maxabs5 = np.max(np.abs(vm[4]))
        maxabs = max(maxabs5, maxabs)

        ### Plot region 6
        maxabs6 = np.max(np.abs(vm[5]))
        maxabs = max(maxabs6, maxabs)
        fig = plt.figure(4)

        colorline(z1, y1, vm[0], cmap=plt.get_cmap('jet'),
                  norm=plt.Normalize(-0, maxabs), linewidth=2)
        colorline(z2, y2, vm[1], cmap=plt.get_cmap('jet'),
                  norm=plt.Normalize(-0, maxabs), linewidth=2)
        colorline(z3, y3, vm[2], cmap=plt.get_cmap('jet'),
                  norm=plt.Normalize(-0, maxabs), linewidth=2)
        colorline(z4, y4, vm[3], cmap=plt.get_cmap('jet'),
                  norm=plt.Normalize(-0, maxabs), linewidth=2)
        colorline(z5, y5, vm[4], cmap=plt.get_cmap('jet'),
                  norm=plt.Normalize(-0, maxabs), linewidth=2)
        colorline(z6, y6, vm[5], cmap=plt.get_cmap('jet'),
                  norm=plt.Normalize(-0, maxabs), linewidth=2)

        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('jet'),
                                   norm=plt.Normalize(-0, maxabs))
        sm.set_array([])
        plt.colorbar(sm, label=r'$\sigma_{vm}$ N/m$^2$', fraction=0.20, pad=0.04,
                     orientation="horizontal")
        plt.xlim(-self.ail.Ca - 0.1, 0.1)
        plt.ylim(-self.ail.ha / 2 - 0.02, self.ail.ha / 2 + 0.02)
        plt.axis('scaled')
        plt.xlabel(r'$-z$ [m]')
        plt.ylabel(r'$y$ [m]')
        if x is not None:
            plt.title(f'Numerical Von Mises stress distribution at {x}m span-wise')
        else:
            plt.title(f'Numerical Von Mises stress distribution')
        plt.show()

    def find_max_stress(self):
        """
        Find the maximum von mises stresses experienced by the cross-section
        by iterating over a list of nodes along the span and finding the max.
        """
        nodes = np.linspace(0.1, self.ail.la-0.1, 500)
        max_stress = 0
        j = 0
        for i in range(len(nodes)):
            _, _, vm = self.compute_stresses(x=nodes[i])
            max_stress_curr = abs(max(vm.min(), vm.max(),key=abs))
            if max_stress_curr > max_stress:
                max_stress = max_stress_curr
                j=i

        print(f"Max stress of {max_stress/1e6}MPa at {nodes[j]}m")

if __name__ == "__main__":
    Aileron = Structure(**section_params_dict)
    load_case = Loading(Aileron, datafile, **loading_params_dict)
    load_case.plot_crossection(0.5)




