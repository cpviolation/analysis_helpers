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
