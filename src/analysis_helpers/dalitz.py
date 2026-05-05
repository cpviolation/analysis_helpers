import math
import numpy as np

def cos_helicity_angle_dalitz(m2ab, m2bc, md, ma, mb, mc):
    """Calculate cosine of helicity angle for a set of two Dalitz plot variables

    :param m2ab: 1st Dalitz plot variable (squared invariant mass of the AB combination)
    :param m2bc: 2nd Dalitz plot variable (squared invariant mass of the BC combination)
    :param md: Mass of the decaying particle D
    :param ma: Mass of the decay product A
    :param mb: Mass of the decay product B
    :param mc: Mass of the decay product C
    :returns: Cosine of the helicity angle of the AB combination
              (of the angle between AB direction in D rest frame
               and A direction in AB rest frame).
    """
    md2 = md ** 2
    ma2 = ma ** 2
    mb2 = mb ** 2
    mc2 = mc ** 2
    #m2ac = md2 + ma2 + mb2 + mc2 - m2ab - m2bc
    mab = np.sqrt(m2ab)
    #mac = np.sqrt(m2ac)
    #mbc = np.sqrt(m2bc)
    #p2a = 0.25 / md2 * (md2 - (mbc + ma) ** 2) * (md2 - (mbc - ma) ** 2)
    #p2b = 0.25 / md2 * (md2 - (mac + mb) ** 2) * (md2 - (mac - mb) ** 2)
    #p2c = 0.25 / md2 * (md2 - (mab + mc) ** 2) * (md2 - (mab - mc) ** 2)
    eb = (m2ab - ma2 + mb2) / 2.0 / mab
    ec = (md2 - m2ab - mc2) / 2.0 / mab
    pb = np.sqrt(eb ** 2 - mb2)
    pc = np.sqrt(ec ** 2 - mc2)
    e2sum = (eb + ec) ** 2
    m2bc_max = e2sum - (pb - pc) ** 2
    m2bc_min = e2sum - (pb + pc) ** 2
    return (m2bc_max + m2bc_min - 2.0 * m2bc) / (m2bc_max - m2bc_min)


def two_body_momentum_squared(md, ma, mb):
    """Squared momentum of two-body decay products D->AB in the D rest frame

    :param md:
    :param ma:
    :param mb:

    """
    return (md ** 2 - (ma + mb) ** 2) * (md ** 2 - (ma - mb) ** 2) / (4 * md ** 2)


def two_body_momentum(md, ma, mb):
    """Momentum of two-body decay products D->AB in the D rest frame

    :param md:
    :param ma:
    :param mb:

    """
    return np.sqrt(two_body_momentum_squared(md, ma, mb))


def vector(x, y, z):
    """
    Make a 3-vector from components. Components are stacked along the last index.

    :param x: x-component of the vector
    :param y: y-component of the vector
    :param z: z-component of the vector
    :returns: 3-vector
    """
    return np.stack([x, y, z], axis=-1)


def lorentz_vector(space, time):
    """
    Make a Lorentz vector from spatial and time components

    :param space: 3-vector of spatial components
    :param time: time component
    :returns: Lorentz vector

    """
    return np.concat([space, np.stack([time], axis=-1)], axis=-1)


class DalitzKinematics:
    def __init__(self, mother_mass, daughters_masses):
        """A class to calculate the kinematics of a decay

        Args:
            mother_mass (float): the mass of the mother particle
            daughters_masses (list): the masses of the daughters particles

        Raises:
            ValueError: Less than two daughters are provided
        """
        if len(daughters_masses) != 3:
            raise ValueError('Three daughters are required in a Dalitz plot')
        self.mother_mass = mother_mass
        self.daughters_masses = daughters_masses
        return

    def otherDaughter(self, daughters_indices):
        """Get the index of the daughter that is not in the list

        Args:
            daughters_indices (list): list of indices of the daughters of interest

        Raises:
            ValueError: list of daughters indices is not of length 2

        Returns:
            int: the index of the daughter that is not in the list
        """
        if len(daughters_indices) != 2:
            raise ValueError('Two daughters indices are required')
        other_indices = np.arange(len(self.daughters_masses))
        other_indices = np.setdiff1d(other_indices, daughters_indices)
        return other_indices[0]

    def mSqMin(self, daughters_indices):
        """Calculate the minimum invariant mass squared of a combination of daughters

        Args:
            daughters_indices (list): the indices of the daughters of interest

        Raises:
            ValueError: no daughters indices are provided

        Returns:
            float: the minimum invariant mass squared of the combination
        """
        if len(daughters_indices) != 2:
            raise ValueError('Two daughters indices are required')
        comb_mass = 0
        for i in daughters_indices:
            comb_mass += self.daughters_masses[i]
        return (comb_mass)**2

    def mSqMax(self, daughters_indices):
        """Calculate the maximum invariant mass squared of a combination of daughters

        Args:
            daughters_indices (list): the indices of the daughters of interest

        Raises:
            ValueError: no daughters indices are provided

        Returns:
            float: the maximum invariant mass squared of the combination
        """
        if len(daughters_indices) != 2:
            raise ValueError('Two daughters indices are required')
        return (self.mother_mass - self.daughters_masses[self.otherDaughter(daughters_indices)])**2

    def EStar2(self, inv_mass, daughters_indices):
        """Calculates the energy squared of a combination of daughters

        Args:
            inv_mass (float): the invariant mass of the combination
            daughters_indices (list): the list of indices of the daughters of interest

        Raises:
            ValueError: two daughters indices are required

        Returns:
            float: the energy squared of the combination to calculate the Dalitz plot range
        """
        if len(daughters_indices) != 2:
            raise ValueError('Two daughters indices are required')
        m1Sq = self.daughters_masses[daughters_indices[0]]**2
        m2Sq = self.daughters_masses[daughters_indices[1]]**2
        Est2 = inv_mass*inv_mass - m1Sq + m2Sq
        Est2/= 2*inv_mass
        return Est2

    def EStar3(self, inv_mass, daughters_indices):
        """Calculates the energy squared of a combination of daughters

        Args:
            inv_mass (float): the invariant mass of the combination
            daughters_indices (list): the list of indices of the daughters of interest

        Raises:
            ValueError: two daughters indices are required

        Returns:
            float: the energy squared of the decay remnants to calculate the Dalitz plot range
        """
        if len(daughters_indices) != 2:
            raise ValueError('Two daughters indices are required')
        mMSq = self.mother_mass**2
        mOtherSq = self.daughters_masses[self.otherDaughter(daughters_indices)]**2
        Est3 = mMSq - inv_mass*inv_mass - mOtherSq
        Est3/= 2*inv_mass
        return Est3

    def OtherMassSq(self, s1, s2):
        """Calculate the invariant mass squared of the remaining combination of daughters

        Args:
            s1 (float): the invariant mass of the first combination
            s2 (float): the invariant mass of the second combination
        """
        s3 = self.mother_mass**2 - s1 - s2
        for m in self.daughters_masses:
            s3 += m**2
        return s3

    def Contour(self, daughters_indices, num_points=1000):
        """Calculate the Dalitz plot contour as mSq23 vs mSq12

        Args:
            daughters_indices (list): the list of indices of the daughters on the interest
            Npoints (int): the number of points for the contour

        Raises:
            ValueError: two daughters indices are required

        Returns:
            tuple: the x and y_low,high coordinates of the contour
        """
        if len(daughters_indices) != 2:
            raise ValueError('Two daughters indices are required')
        # m1Sq = self.daughters_masses[daughters_indices[0]]**2
        m2Sq = self.daughters_masses[daughters_indices[1]]**2
        m3Sq = self.daughters_masses[self.otherDaughter(daughters_indices)]**2
        mSqMin = self.mSqMin(daughters_indices)
        mSqMax = self.mSqMax(daughters_indices)
        mSq = np.linspace(mSqMin, mSqMax, num_points)
        Est2 = self.EStar2(np.sqrt(mSq), daughters_indices)
        Est3 = self.EStar3(np.sqrt(mSq), daughters_indices)
        Ymin = np.power(Est2+Est3, 2) - np.power(np.sqrt(Est2*Est2-m2Sq) + np.sqrt(Est3*Est3-m3Sq),2)
        Ymax = np.power(Est2+Est3, 2) - np.power(np.sqrt(Est2*Est2-m2Sq) - np.sqrt(Est3*Est3-m3Sq),2)
        return (mSq, Ymin, Ymax)


class DalitzPhaseSpace:
    """
    Class for Dalitz plot (2D) phase space for the 3-body decay D->ABC
    """

    def __init__(
        self,
        ma,
        mb,
        mc,
        md,
        mabrange=None,
        mbcrange=None,
        macrange=None,
        symmetric=False,
    ):
        """
        Constructor
          ma - A mass
          mb - B mass
          mc - C mass
          md - D (mother) mass
        """
        self.ma = ma
        self.mb = mb
        self.mc = mc
        self.md = md
        self.ma2 = ma * ma
        self.mb2 = mb * mb
        self.mc2 = mc * mc
        self.md2 = md * md
        self.msqsum = self.md2 + self.ma2 + self.mb2 + self.mc2
        self.minab = (ma + mb) ** 2
        self.maxab = (md - mc) ** 2
        self.minbc = (mb + mc) ** 2
        self.maxbc = (md - ma) ** 2
        self.minac = (ma + mc) ** 2
        self.maxac = (md - mb) ** 2
        self.macrange = macrange
        self.symmetric = symmetric
        self.min_mprimeac = 0.0
        self.max_mprimeac = 1.0
        self.min_thprimeac = 0.0
        self.max_thprimeac = 1.0
        if self.symmetric:
            self.max_thprimeac = 0.5
        if mabrange:
            if mabrange[1] ** 2 < self.maxab:
                self.maxab = mabrange[1] ** 2
            if mabrange[0] ** 2 > self.minab:
                self.minab = mabrange[0] ** 2
        if mbcrange:
            if mbcrange[1] ** 2 < self.maxbc:
                self.maxbc = mbcrange[1] ** 2
            if mbcrange[0] ** 2 > self.minbc:
                self.minbc = mbcrange[0] ** 2


    def inside(self, x):
        """
        Check if the point x=(m2ab, m2bc) is inside the phase space
        """
        m2ab = self.m2ab(x)
        m2bc = self.m2bc(x)
        mab = np.sqrt(m2ab)

        inside = np.logical_and(
            np.logical_and(np.greater(m2ab, self.minab), np.less(m2ab, self.maxab)),
            np.logical_and(np.greater(m2bc, self.minbc), np.less(m2bc, self.maxbc)),
        )

        if self.macrange:
            m2ac = self.msqsum - m2ab - m2bc
            inside = np.logical_and(
                inside,
                np.logical_and(
                    np.greater(m2ac, self.macrange[0] ** 2),
                    np.less(m2ac, self.macrange[1] ** 2),
                ),
            )

        if self.symmetric:
            inside = np.logical_and(inside, np.greater(m2bc, m2ab))

        eb = (m2ab - self.ma2 + self.mb2) / 2.0 / mab
        ec = (self.md2 - m2ab - self.mc2) / 2.0 / mab
        p2b = eb ** 2 - self.mb2
        p2c = ec ** 2 - self.mc2
        inside = np.logical_and(
            inside, np.logical_and(np.greater(p2c, 0), np.greater(p2b, 0))
        )
        pb = np.sqrt(p2b)
        pc = np.sqrt(p2c)
        e2bc = (eb + ec) ** 2
        m2bc_max = e2bc - (pb - pc) ** 2
        m2bc_min = e2bc - (pb + pc) ** 2
        return np.logical_and(
            inside, np.logical_and(np.greater(m2bc, m2bc_min), np.less(m2bc, m2bc_max))
        )

    
    def filter(self, x):
        return x[self.inside(x)]

    #
    def unfiltered_sample(self, size, maximum=None):
        """
        Return TF graph for uniform sample of point within phase space.
          size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                     so the number of points in the output will be <size.
          majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                     uniform number from 0 to majorant. Useful for accept-reject toy MC.
        """
        v = [
            np.random.uniform(self.minab, self.maxab, size),
            np.random.uniform(self.minbc, self.maxbc, size),
        ]

        if maximum is not None:
            v += [np.random.uniform(0.0, maximum, size)]
        return np.stack(v, axis=1)

    #
    def uniform_sample(self, size, maximum=None):
        """
        Generate uniform sample of point within phase space.
          size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                     so the number of points in the output will be <size.
          majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                     uniform number from 0 to majorant. Useful for accept-reject toy MC.
        Note it does not actually generate the sample, but returns the data flow graph for generation,
        which has to be run within TF session.
        """
        return self.filter(self.unfiltered_sample(size, maximum))

    
    def rectangular_grid_sample(self, size_ab, size_bc, space_to_sample="DP"):
        """
        Create a data sample in the form of rectangular grid of points within the phase space.
        Useful for normalisation.
          size_ab : number of grid nodes in m2ab range
          size_bc : number of grid nodes in m2bc range
          space_to_sample: Sampling is done according to cases below but all of them return DP vars (m^2_{ab}, m^2_{bc}).
              -if 'DP': Unifrom sampling is in (m^2_{ab}, m^2_{bc})
              -if 'linDP': Samples in (m_{ab}, m_{bc})
              -if 'sqDP': Samples in (mPrimeAC, thPrimeAC).
        """
        size = size_ab * size_bc
        mgrid = np.lib.index_tricks.nd_grid()
        if space_to_sample == "linDP":
            vab = (
                mgrid[0:size_ab, 0:size_bc][0]
                * (math.sqrt(self.maxab) - math.sqrt(self.minab))
                / float(size_ab)
                + math.sqrt(self.minab)
            ) ** 2.0
            vbc = (
                mgrid[0:size_ab, 0:size_bc][1]
                * (math.sqrt(self.maxbc) - math.sqrt(self.minbc))
                / float(size_bc)
                + math.sqrt(self.minbc)
            ) ** 2.0
            v = [vab.reshape(size).astype("d"), vbc.reshape(size).astype("d")]
            dlz = np.stack(v, axis=1)
        elif space_to_sample == "sqDP":
            x = np.linspace(self.min_mprimeac, self.max_mprimeac, size_ab)
            y = np.linspace(self.min_thprimeac, self.max_thprimeac, size_bc)
            # Remove corners of sqDP as they lie outside phsp
            xnew = x[
                (x > self.min_mprimeac)
                & (x < self.max_mprimeac)
                & (y > self.min_thprimeac)
                & (y < self.max_thprimeac)
            ]
            ynew = y[
                (x > self.min_mprimeac)
                & (x < self.max_mprimeac)
                & (y > self.min_thprimeac)
                & (y < self.max_thprimeac)
            ]
            mprimeac, thprimeac = np.meshgrid(xnew, ynew)
            dlz = self.from_square_dalitz_plot(
                mprimeac.flatten().astype("d"), thprimeac.flatten().astype("d")
            )
        else:
            vab = (
                mgrid[0:size_ab, 0:size_bc][0]
                * (self.maxab - self.minab)
                / float(size_ab)
                + self.minab
            )
            vbc = (
                mgrid[0:size_ab, 0:size_bc][1]
                * (self.maxbc - self.minbc)
                / float(size_bc)
                + self.minbc
            )
            v = [vab.reshape(size).astype("d"), vbc.reshape(size).astype("d")]
            dlz = np.stack(v, axis=1)

        return self.filter(dlz)

    
    def m2ab(self, sample):
        """
        Return m2ab variable (vector) for the input sample
        """
        return sample[..., 0]

    
    def m2bc(self, sample):
        """
        Return m2bc variable (vector) for the input sample
        """
        return sample[..., 1]

    
    def m2ac(self, sample):
        """
        Return m2ac variable (vector) for the input sample.
        It is calculated from m2ab and m2bc
        """
        return self.msqsum - self.m2ab(sample) - self.m2bc(sample)

    
    def cos_helicity_ab(self, sample):
        """
        Calculate cos(helicity angle) of the AB resonance
        """
        return cos_helicity_angle_dalitz(
            self.m2ab(sample), self.m2bc(sample), self.md, self.ma, self.mb, self.mc
        )

    
    def cos_helicity_bc(self, sample):
        """
        Calculate cos(helicity angle) of the BC resonance
        """
        return cos_helicity_angle_dalitz(
            self.m2bc(sample), self.m2ac(sample), self.md, self.mb, self.mc, self.ma
        )

    
    def cos_helicity_ac(self, sample):
        """
        Calculate cos(helicity angle) of the AC resonance
        """
        return cos_helicity_angle_dalitz(
            self.m2ac(sample), self.m2ab(sample), self.md, self.mc, self.ma, self.mb
        )

    
    def m_prime_ac(self, sample):
        """
        Square Dalitz plot variable m'
        """
        mac = np.sqrt(self.m2ac(sample))
        return (
            np.arccos(
                2
                * (mac - math.sqrt(self.minac))
                / (math.sqrt(self.maxac) - math.sqrt(self.minac))
                - 1.0
            )
            / math.pi
        )

    
    def theta_prime_ac(self, sample):
        """
        Square Dalitz plot variable theta'
        """
        return np.arccos(self.cos_helicity_ac(sample)) / math.pi

    
    def m_prime_ab(self, sample):
        """
        Square Dalitz plot variable m'
        """
        mab = np.sqrt(self.m2ab(sample))
        return (
            np.arccos(
                2
                * (mab - math.sqrt(self.minab))
                / (math.sqrt(self.maxab) - math.sqrt(self.minab))
                - 1.0
            )
            / math.pi
        )

    
    def from_square_dalitz_plot(self, mprimeac, thprimeac):
        """
        sample: Given mprimeac and thprimeac, returns 2D tensor for (m2ab, m2bc).
        Make sure you don't pass in sqDP corner points as they lie outside phsp.
        """
        m2AC = (
            0.25
            * (
                self.maxac ** 0.5 * np.cos(math.pi * mprimeac)
                + self.maxac ** 0.5
                - self.minac ** 0.5 * np.cos(math.pi * mprimeac)
                + self.minac ** 0.5
            )
            ** 2
        )
        m2AB = (
            0.5
            * (
                -(m2AC ** 2)
                + m2AC * self.ma ** 2
                + m2AC * self.mb ** 2
                + m2AC * self.mc ** 2
                + m2AC * self.md ** 2
                - m2AC
                * np.sqrt(
                    (
                        m2AC * (m2AC - 2.0 * self.ma ** 2 - 2.0 * self.mc ** 2)
                        + self.ma ** 4
                        - 2.0 * self.ma ** 2 * self.mc ** 2
                        + self.mc ** 4
                    )
                    / m2AC
                )
                * np.sqrt(
                    (
                        m2AC * (m2AC - 2.0 * self.mb ** 2 - 2.0 * self.md ** 2)
                        + self.mb ** 4
                        - 2.0 * self.mb ** 2 * self.md ** 2
                        + self.md ** 4
                    )
                    / m2AC
                )
                * np.cos(math.pi * thprimeac)
                - self.ma ** 2 * self.mb ** 2
                + self.ma ** 2 * self.md ** 2
                + self.mb ** 2 * self.mc ** 2
                - self.mc ** 2 * self.md ** 2
            )
            / m2AC
        )
        m2BC = self.msqsum - m2AC - m2AB
        return np.stack([m2AB, m2BC], axis=1)

    
    def square_dalitz_plot_jacobian(self, sample):
        """
        sample: [mAB^2, mBC^2]
        Return the jacobian determinant (|J|) of tranformation from dmAB^2*dmBC^2 -> |J|*dMpr*dThpr where Mpr, Thpr are defined in (AC) frame.
        """
        mPrime = self.m_prime_ac(sample)
        thPrime = self.theta_prime_ac(sample)

        diff_AC = np.float(np.sqrt(self.maxac) - np.sqrt(self.minac))
        mAC = np.const(0.5) * diff_AC * (
            np.const(1.0) + np.cos(np.pi() * mPrime)
        ) + np.float(np.sqrt(self.minac))
        mACSq = mAC * mAC

        eAcmsAC = (
            np.const(0.5)
            * (
                mACSq
                - np.float(self.mc2)
                + np.float(self.ma2)
            )
            / mAC
        )
        eBcmsAC = (
            np.const(0.5)
            * (
                np.float(self.md) ** 2.0
                - mACSq
                - np.float(self.mb2)
            )
            / mAC
        )

        pAcmsAC = np.sqrt(eAcmsAC ** 2.0 - np.float(self.ma2))
        pBcmsAC = np.sqrt(eBcmsAC ** 2.0 - np.float(self.mb2))

        deriv1 = np.pi * np.const(0.5) * diff_AC * np.sin(np.pi() * mPrime)
        deriv2 = np.pi * np.sin(np.pi() * thPrime)

        return np.const(4.0) * pAcmsAC * pBcmsAC * mAC * deriv1 * deriv2

    
    def invariant_mass_jacobian(self, sample):
        """
        sample: [mAB^2, mBC^2]
        Return the jacobian determinant (|J|) of tranformation from dmAB^2*dmBC^2 -> |J|*dmAB*dmBC. |J| = 4*mAB*mBC
        """
        return (
            np.const(4.0)
            * np.sqrt(self.m2ab(sample))
            * np.sqrt(self.m2bc(sample))
        )

    
    def theta_prime_ab(self, sample):
        """
        Square Dalitz plot variable theta'
        """
        return np.arccos(-self.cos_helicity_ab(sample)) / math.pi

    
    def m_prime_bc(self, sample):
        """
        Square Dalitz plot variable m'
        """
        mbc = np.sqrt(self.m2bc(sample))
        return (
            np.arccos(
                2
                * (mbc - math.sqrt(self.minbc))
                / (math.sqrt(self.maxbc) - math.sqrt(self.minbc))
                - 1.0
            )
            / math.pi
        )

    
    def theta_prime_bc(self, sample):
        """
        Square Dalitz plot variable theta'
        """
        return np.arccos(-self.cos_helicity_bc(sample)) / math.pi

    
    def from_vectors(self, m2ab, m2bc):
        """
        Create Dalitz plot tensor from two vectors of variables, m2ab and m2bc
        """
        return np.stack([m2ab, m2bc], axis=1)

    
    def final_state_momenta(self, m2ab, m2bc):
        """
        Calculate 4-momenta of final state tracks in a certain reference frame
        (decay is in x-z plane, particle A moves along z axis)
          m2ab, m2bc : invariant masses of AB and BC combinations
        """

        m2ac = self.msqsum - m2ab - m2bc

        p_a = two_body_momentum(self.md, self.ma, np.sqrt(m2bc))
        p_b = two_body_momentum(self.md, self.mb, np.sqrt(m2ac))
        p_c = two_body_momentum(self.md, self.mc, np.sqrt(m2ab))

        cos_theta_b = (p_a * p_a + p_b * p_b - p_c * p_c) / (2.0 * p_a * p_b)
        cos_theta_c = (p_a * p_a + p_c * p_c - p_b * p_b) / (2.0 * p_a * p_c)

        p4a = lorentz_vector(
            vector(np.zeros(p_a), np.zeros(p_a), p_a),
            np.sqrt(p_a ** 2 + self.ma2),
        )
        p4b = lorentz_vector(
            vector(
                p_b * np.sqrt(1.0 - cos_theta_b ** 2),
                np.zeros(p_b),
                -p_b * cos_theta_b,
            ),
            np.sqrt(p_b ** 2 + self.mb2),
        )
        p4c = lorentz_vector(
            vector(
                -p_c * np.sqrt(1.0 - cos_theta_c ** 2),
                np.zeros(p_c),
                -p_c * cos_theta_c,
            ),
            np.sqrt(p_c ** 2 + self.mc2),
        )
        return (p4a, p4b, p4c)

    def dimensionality(self):
        return 2

    def bounds(self) :
        return [(self.minab, self.maxab), (self.minbc, self.maxbc)]
