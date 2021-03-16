from params import *
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.integrate import quad, dblquad

"""
    Filename: loading.py
    Author: Alex Shokolarov, Remco Roelofs
    Date last modified: 26/02/2021
    Python version: 3.8
"""

class Stiffener:

    def __init__(self, z, y, tst, hst, wst):
        """
        Constructor for stiffener class
            z - z coordinate of stiffener
            y - y coordinate of stiffener
            tst - thickness of stiffener
            hst - height of stiffener
            wst - width of stiffener
        """
        self.coor = [z, y]
        self.A = (hst + wst) * tst

    def __str__(self):
        return f"Stiffener at {[self.coor[0], self.coor[1]]}"

class Structure:

    def __init__(self, **kwargs):
        """
        Constructors used to initialize a structure from the input parameters.
        Parameters:
            Ca - chord length [m]
            la - span [m]
            x1 - location of hinge 1 [m]
            x2 - location of hinge 2 [m]
            x3 - location of hinge 3 [m]
            xa - distance between actuators [m]
            ha - aileron height [m]
            tsk - skin thickness [m]
            tsp - spar thickness [m]
            tst - stiffener thickness [m]
            hst - stiffener height [m]
            wst - stiffener width
            nst - number of stiffeners
        """

        self.Ca = kwargs["Ca"]
        self.la = kwargs["la"]
        self.x1 = kwargs["x1"]
        self.x2 = kwargs["x2"]
        self.x3 = kwargs["x3"]
        self.xa = kwargs["xa"]
        self.ha = kwargs["ha"]
        self.tsk = kwargs["tsk"]
        self.tsp = kwargs["tsp"]
        self.tst = kwargs["tst"]
        self.hst = kwargs["hst"]
        self.wst = kwargs["wst"]
        self.nst = kwargs["nst"]
        self.E = kwargs["E"]
        self.G = kwargs["G"]

        self.td = self.Ca - self.ha / 2  # horizontal distance of triangular section
        self.psi = np.arctan2(self.ha / 2, self.td)  # inclination angle of skin about z-axis
        self.sl = self.td / np.cos(self.psi)  # skin length
        self.stiffeners = [] # Array holding all stiffeners
        self.elements = {} # Dictionary to store elements along with their centroid and area

        self._check_valid_input()
        self._emplace_stiffeners()


    def _check_valid_input(self):
        # check if input data is valid
        if self.nst % 2 != 1:
            raise Exception("Number of stringers is even number.")

    def _emplace_stiffeners(self):
        # position the stiffeners on the cross-section

        # place stiffener on leading edge
        self.stiffeners.append(Stiffener(0, 0, self.tst, self.hst, self.wst))

        P = np.pi * self.ha / 2 + 2 * self.sl # compute perimeter
        dst = P / self.nst # find stiffener spacing
        phi = 2 * dst / self.ha # angle between stiffeners on semi-circle

        ncirc = int( (2 * (self.ha * np.pi / 4 / dst) ) / 2 ) # num of stiffeners placed on semi-circle
        for i in range(1, ncirc + 1):
            # find coordinates of stiffeners on semi-circle and place them
            z = (self.ha/2) * (1 - np.cos(i * phi))
            y = (self.ha/2) * (np.sin(i * phi))
            # place one on y and one one -y coordinate
            self.stiffeners.append(Stiffener(z,  y, self.tst, self.hst, self.wst))
            self.stiffeners.append(Stiffener(z, -y, self.tst, self.hst, self.wst))

        nskin = int((self.nst-1)/2) - ncirc # num of stiffeners on triangular section
        for i in range(nskin):
            # find coordinates of stiffeners on triangular section and place them
            z = self.Ca - (dst/2)*np.cos(self.psi) - i*dst*np.cos(self.psi)
            y = (dst/2)*np.sin(self.psi) + i*dst*np.sin(self.psi)
            # place one on y and one one -y coordinate
            self.stiffeners.append(Stiffener(z, y, self.tst, self.hst, self.wst))
            self.stiffeners.append(Stiffener(z, -y, self.tst, self.hst, self.wst))

    def _compute_centroid_elements(self):
        # Determine the areas and centroid for the different elements

        # Semi-circle centroid
        zcirc = self.ha * (0.5 - 1/np.pi)
        ycirc = 0
        acirc = np.pi * (self.ha/2) * self.tsk
        self.elements['SEMI-CIRCLE'] = [zcirc, ycirc, acirc]

        # Semi-circle spar
        zspar = self.ha/2
        yspar = 0
        aspar = self.ha * self.tsp
        self.elements['SPAR'] = [zspar, yspar, aspar]

        # Skin 1
        zt1 = self.ha/2 + self.td/2
        yt1 = self.ha/4
        at1 = self.sl * self.tsk
        self.elements['SKIN1'] = [zt1, yt1, at1]

        # Skin 2
        zt2 = self.ha / 2 + self.td / 2
        yt2 = -self.ha / 4
        at2 = self.sl * self.tsk
        self.elements['SKIN2'] = [zt2, yt2, at2]

    @property
    def centroid(self):
        """
        Compute the centroid of the cross-section
        :return: [zbar, ybar]
        """
        self._compute_centroid_elements()

        # Compute total area of cross-section
        A_TOTAL = self.elements['SEMI-CIRCLE'][2] + self.elements['SPAR'][2] \
                   + self.elements['SKIN1'][2] + self.elements['SKIN2'][2]
        for s in self.stiffeners:
            A_TOTAL += s.A

        # Compute weighted area wrt distance
        WEIGHTED_A = self.elements['SEMI-CIRCLE'][2] * self.elements['SEMI-CIRCLE'][0] \
                    + self.elements['SPAR'][2] * self.elements['SPAR'][0] \
                    + self.elements['SKIN1'][2] * self.elements['SKIN1'][0] \
                    + self.elements['SKIN2'][2] * self.elements['SKIN2'][0] \

        for s in self.stiffeners:
            WEIGHTED_A += s.A * s.coor[0]

        zbar = WEIGHTED_A / A_TOTAL

        return [zbar, 0]


    @property
    def SMOA(self):
        """
        Compute second moment of area of cross-section around z and y axes
        :return: [Izz, Iyy]
        """
        centroid = self.centroid
        Izz, Iyy = 0, 0

        # Semi-circle
        Izz_circ = (np.pi / 2) * (self.ha / 2) ** 3 * self.tsk
        Iyy_circ = Izz_circ - (self.ha / 2 - self.elements['SEMI-CIRCLE'][0]) ** 2 \
                   * (self.ha * np.pi / 2) * self.tsk
        Izz += Izz_circ
        Iyy += Iyy_circ + self.elements['SEMI-CIRCLE'][2] * (centroid[0] - self.elements['SEMI-CIRCLE'][0]) \
               * (centroid[0] - self.elements['SEMI-CIRCLE'][0])

        # Spar
        Izz_spar = (self.ha ** 3 * self.tsp) / 12
        Iyy_spar = (self.elements['SPAR'][0] - centroid[0]) ** 2 * self.elements['SPAR'][2]
        Izz += Izz_spar
        Iyy += Iyy_spar

        # Skin
        Izz_skin = self.tsk * self.sl ** 3 * np.sin(self.psi) ** 2 / 12
        Iyy_skin = self.tsk * self.sl ** 3 * np.cos(self.psi) ** 2 / 12

        Izz_skin += self.elements['SKIN1'][2] * (centroid[1] - self.elements['SKIN1'][1]) ** 2
        Iyy_skin += self.elements['SKIN1'][2] * (centroid[0] - self.elements['SKIN1'][0]) ** 2

        Izz += Izz_skin * 2
        Iyy += Iyy_skin * 2

        # Stiffeners
        for s in self.stiffeners:
            Izz += s.A * (centroid[1] - s.coor[1]) ** 2
            Iyy += s.A * (centroid[0] - s.coor[0]) ** 2
        return Izz, Iyy

    @property
    def shear_center(self):
        """
        Compute the shear center of the section by applying a unit shear force in y,
        calculating the shear distribution and the moments it causes.
        :return: [zbar, ybar]
        """
        Izz, _ = self.SMOA

        r = self.ha / 2

        A = sum([stiff.A for stiff in self.stiffeners])
        P = np.pi * self.ha / 2 + 2 * self.sl
        t_add = A/P
        tsk = self.tsk + t_add
        tsp = self.tsp

        Sy = 1

        qb1 = lambda x: (Sy*tsk*r*r/Izz)*(np.cos(x) - 1)
        qb2 = lambda x: -(Sy*tsp/2/Izz) * x * x
        qb3 = lambda x: -(Sy*tsk/Izz)*(r*x - (r/2/self.sl)*x*x) + qb1(np.pi/2) + qb2(r)
        qb4 = lambda x: (Sy*tsk/2/Izz)*(r*x*x/self.sl) + qb3(self.sl)
        qb5 = lambda x: -(Sy*tsp/2/Izz) * x * x
        qb6 = lambda x: (Sy*tsk*r*r/Izz)*np.cos(x) + qb4(self.sl) - qb5(-r)

        A = np.zeros((2,2))
        A[0,0] = r*np.pi/tsk + 2*r/tsp
        A[0,1] = -2*r/tsp
        A[1,0] = -2*r/tsp
        A[1,1] = 2*self.sl/tsk + 2*r/tsp

        b = np.zeros(2)
        qb1_int = lambda x: qb1(x) * r / tsk
        qb2_int = lambda x: qb2(x) / tsp
        qb6_int = lambda x: qb6(x) * r / tsk
        b[0] = quad(qb1_int,0,np.pi/2)[0] - quad(qb2_int,0,r)[0] - quad(qb2_int,0,-r)[0] + quad(qb6_int,-np.pi/2,0)[0]
        qb3_int = lambda x: qb3(x) / tsk
        qb4_int = lambda x: qb4(x) / tsk
        qb5_int = lambda x: qb5(x) / tsp
        b[1] = quad(qb2_int,0,r)[0] + quad(qb3_int,0,self.sl)[0] + quad(qb4_int,0,self.sl)[0] + quad(qb5_int,0,-r)[0]

        qr0, qr1 = np.linalg.solve(A,-b)

        q1 = lambda x: qb1(x) + qr0
        q3 = lambda x: qb3(x) + qr1
        q4 = lambda x: qb4(x) + qr1
        q6 = lambda x: qb6(x) + qr0

        m1 = lambda x: q1(x)*r*r
        m3 = lambda x: q3(x)*self.sl*np.sin(self.psi)
        m4 = lambda x: q4(x)*self.sl*np.sin(self.psi)
        m6 = lambda x: q6(x)*r*r

        M = quad(m1,0,np.pi/2)[0] + quad(m3,0,self.sl)[0] + quad(m4,0,self.sl)[0] + quad(m6,-np.pi/2,0)[0]
        z_sc = -M / Sy

        return r - z_sc

    @property
    def torsional_stiffness(self):
        """
        Calculate torsional stiffness from Torsion equation and twist compatibility equations.
        :return J: torsional stiffness
        """
        r = self.ha/2
        A1 = np.pi * r**2 / 2
        A2 = (self.Ca - r) * r
        tsk = self.tsk
        tsp = self.tsp

        A = np.zeros((3,3))
        A[0,0] = 2 * A1
        A[0,1] = 2 * A2
        A[1,0] = (r * np.pi / tsk + 2 * r / tsp) / (2*A1)
        A[1,1] = (-2 * r / tsp) / (2*A1)
        A[1,2] = -1
        A[2,0] = (-2 * r / tsp) / (2*A2)
        A[2,1] = (2 * self.sl / tsk + 2 * r / tsp) / (2*A2)
        A[2,2] = -1

        b = np.zeros(3)
        b[0] = 1
        b[1] = 0
        b[2] = 0

        x = np.linalg.solve(A,b)
        return 1 / x[2]

    def plot_cross_section(self):
        # Plot the cross-section of the aileron
        fig, ax = plt.subplots()

        # plot semi-circle
        semi_circle = patches.Arc([self.ha/2, 0], self.ha, self.ha, theta1=r2d*np.pi/2, theta2=3/2*r2d*np.pi, color='k', linewidth=2)
        ax.add_patch(semi_circle)

        # plot spar
        ax.plot([self.ha/2, self.ha/2], [self.ha/2, -self.ha/2], color='k', linewidth=2)

        # plot triangular section
        ax.plot([self.ha/2, self.ha/2 + self.td], [ self.ha/2, 0], color='k', linewidth=2)
        ax.plot([self.ha/2, self.ha/2 + self.td], [-self.ha/2, 0], color='k', linewidth=2)

        # plot stiffeners
        for s in self.stiffeners:
            ax.plot(s.coor[0], s.coor[1], marker='o', color='r')

        # plot centroid
        centroid = self.centroid
        ax.scatter(centroid[0], centroid[1] , marker='x', label='centroid')

        # plot shear center
        shear_center = self.shear_center
        ax.scatter(shear_center, 0, marker='x', label='shear center')
        plt.xlabel("Z coordinate[m]")
        plt.ylabel("Y coordinate[m]")
        plt.title("Cross-section of the aileron.")
        plt.legend()
        plt.show()


# Testing section
if __name__ == "__main__":
    Aileron = Structure(**section_params_dict)
    


