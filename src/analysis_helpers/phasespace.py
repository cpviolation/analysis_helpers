"""Pure-NumPy phase-space helpers.

Adapted from (TensorFlow originals):
  DecayTimePhaseSpace:
    https://github.com/cpviolation/tfa_examples/blob/main/develtfa/decaytime_phasespace.py
  CombinedPhaseSpace:
    https://github.com/cpviolation/AmpliTF/blob/master/amplitf/phasespace/combined_phasespace.py
"""

import numpy as np

from .dalitz import cos_helicity_angle_dalitz, lorentz_vector, two_body_momentum, vector

__all__ = ["DalitzPhaseSpace", "DecayTimePhaseSpace", "CombinedPhaseSpace"]


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
          majorant : if supplied, add a 3rd dimension to the generated array which is
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
          majorant : if supplied, add a 3rd dimension to the generated array which is
                     uniform number from 0 to majorant. Useful for accept-reject toy MC.
        The returned array is generated immediately using NumPy.
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
        grid_ab, grid_bc = np.meshgrid(
            np.arange(size_ab), np.arange(size_bc), indexing="ij"
        )
        if space_to_sample == "linDP":
            vab = (
                grid_ab
                * (np.sqrt(self.maxab) - np.sqrt(self.minab))
                / float(size_ab)
                + np.sqrt(self.minab)
            ) ** 2.0
            vbc = (
                grid_bc
                * (np.sqrt(self.maxbc) - np.sqrt(self.minbc))
                / float(size_bc)
                + np.sqrt(self.minbc)
            ) ** 2.0
            v = [vab.reshape(size).astype("d"), vbc.reshape(size).astype("d")]
            dlz = np.stack(v, axis=1)
        elif space_to_sample == "sqDP":
            x = np.linspace(self.min_mprimeac, self.max_mprimeac, size_ab)[1:-1]
            y = np.linspace(self.min_thprimeac, self.max_thprimeac, size_bc)[1:-1]
            mprimeac, thprimeac = np.meshgrid(x, y, indexing="ij")
            dlz = self.from_square_dalitz_plot(
                mprimeac.flatten().astype("d"), thprimeac.flatten().astype("d")
            )
        else:
            vab = (
                grid_ab
                * (self.maxab - self.minab)
                / float(size_ab)
                + self.minab
            )
            vbc = (
                grid_bc
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
                * (mac - np.sqrt(self.minac))
                / (np.sqrt(self.maxac) - np.sqrt(self.minac))
                - 1.0
            )
            / np.pi
        )

    
    def theta_prime_ac(self, sample):
        """
        Square Dalitz plot variable theta'
        """
        return np.arccos(self.cos_helicity_ac(sample)) / np.pi

    
    def m_prime_ab(self, sample):
        """
        Square Dalitz plot variable m'
        """
        mab = np.sqrt(self.m2ab(sample))
        return (
            np.arccos(
                2
                * (mab - np.sqrt(self.minab))
                / (np.sqrt(self.maxab) - np.sqrt(self.minab))
                - 1.0
            )
            / np.pi
        )

    
    def from_square_dalitz_plot(self, mprimeac, thprimeac):
        """
        sample: Given mprimeac and thprimeac, returns 2D tensor for (m2ab, m2bc).
        Make sure you don't pass in sqDP corner points as they lie outside phsp.
        """
        m2AC = (
            0.25
            * (
                self.maxac ** 0.5 * np.cos(np.pi * mprimeac)
                + self.maxac ** 0.5
                - self.minac ** 0.5 * np.cos(np.pi * mprimeac)
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
                * np.cos(np.pi * thprimeac)
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

        diff_AC = np.sqrt(self.maxac) - np.sqrt(self.minac)
        mAC = 0.5 * diff_AC * (1.0 + np.cos(np.pi * mPrime)) + np.sqrt(self.minac)
        mACSq = mAC * mAC

        eAcmsAC = (
            0.5 * (mACSq - self.mc2 + self.ma2) / mAC
        )
        eBcmsAC = (
            0.5 * (self.md2 - mACSq - self.mb2) / mAC
        )

        pAcmsAC = np.sqrt(eAcmsAC ** 2.0 - self.ma2)
        pBcmsAC = np.sqrt(eBcmsAC ** 2.0 - self.mb2)

        deriv1 = np.pi * 0.5 * diff_AC * np.sin(np.pi * mPrime)
        deriv2 = np.pi * np.sin(np.pi * thPrime)

        return 4.0 * pAcmsAC * pBcmsAC * mAC * deriv1 * deriv2

    
    def invariant_mass_jacobian(self, sample):
        """
        sample: [mAB^2, mBC^2]
        Return the jacobian determinant (|J|) of tranformation from dmAB^2*dmBC^2 -> |J|*dmAB*dmBC. |J| = 4*mAB*mBC
        """
        return 4.0 * np.sqrt(self.m2ab(sample)) * np.sqrt(self.m2bc(sample))

    
    def theta_prime_ab(self, sample):
        """
        Square Dalitz plot variable theta'
        """
        return np.arccos(-self.cos_helicity_ab(sample)) / np.pi

    
    def m_prime_bc(self, sample):
        """
        Square Dalitz plot variable m'
        """
        mbc = np.sqrt(self.m2bc(sample))
        return (
            np.arccos(
                2
                * (mbc - np.sqrt(self.minbc))
                / (np.sqrt(self.maxbc) - np.sqrt(self.minbc))
                - 1.0
            )
            / np.pi
        )

    
    def theta_prime_bc(self, sample):
        """
        Square Dalitz plot variable theta'
        """
        return np.arccos(-self.cos_helicity_bc(sample)) / np.pi

    
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


class DecayTimePhaseSpace:
    """1-D decay-time phase space.

    Generates decay times t drawn from an exponential distribution with
    lifetime ``tau`` (t > 0).
    """

    def __init__(self, tau: float = 1.0):
        self.tau = tau

    # ------------------------------------------------------------------
    def dimensionality(self) -> int:
        return 1

    def bounds(self):
        return [(0.0, float("inf"))]

    # ------------------------------------------------------------------
    def inside(self, x: np.ndarray) -> np.ndarray:
        """Boolean mask: True where t > 0.

        ``x`` may be shape (N,) or (N, 1); returns shape (N,).
        """
        t = x[:, 0] if x.ndim == 2 else x
        return t > 0.0

    def filter(self, x: np.ndarray) -> np.ndarray:
        return x[self.inside(x)]

    # ------------------------------------------------------------------
    def unfiltered_sample(self, size: int, maximum=None) -> np.ndarray:
        """Generate *size* points; returns shape (size, 1) [or (size, 2) with majorant].

        Decay times are drawn via inverse-CDF: t = -tau * ln(1-U), U~Uniform(0,1).
        If ``maximum`` is given, an extra uniform(0, maximum) column is appended
        (classic majorant column for accept-reject MC).
        """
        u = np.random.uniform(0.0, 1.0, size)
        t = -self.tau * np.log1p(-u)           # numerically stable
        cols = [t[:, np.newaxis]]
        if maximum is not None:
            cols.append(np.random.uniform(0.0, maximum, (size, 1)))
        return np.concatenate(cols, axis=1)

    def uniform_sample(self, size: int, maximum=None) -> np.ndarray:
        """Filtered sample — all returned points satisfy t > 0 (always true here)."""
        return self.filter(self.unfiltered_sample(size, maximum))

    # ------------------------------------------------------------------
    def t(self, sample: np.ndarray) -> np.ndarray:
        """Extract the decay-time column; returns shape (N,)."""
        return sample[:, 0] if sample.ndim == 2 else sample


# ======================================================================

class CombinedPhaseSpace:
    """Cartesian (direct) product of two phase spaces.

    ``phsp1`` occupies the first ``phsp1.dimensionality()`` columns;
    ``phsp2`` occupies the remaining columns.
    """

    def __init__(self, phsp1, phsp2):
        self.phsp1 = phsp1
        self.phsp2 = phsp2

    # ------------------------------------------------------------------
    def dimensionality(self) -> int:
        return self.phsp1.dimensionality() + self.phsp2.dimensionality()

    def bounds(self):
        return self.phsp1.bounds() + self.phsp2.bounds()

    # ------------------------------------------------------------------
    def data1(self, x: np.ndarray) -> np.ndarray:
        """First phase-space slice (phsp1 columns)."""
        return x[..., : self.phsp1.dimensionality()]

    def data2(self, x: np.ndarray) -> np.ndarray:
        """Second phase-space slice (phsp2 columns)."""
        return x[..., self.phsp1.dimensionality():]

    # ------------------------------------------------------------------
    def inside(self, x: np.ndarray) -> np.ndarray:
        return self.phsp1.inside(self.data1(x)) & self.phsp2.inside(self.data2(x))

    def filter(self, x: np.ndarray) -> np.ndarray:
        return x[self.inside(x)]

    # ------------------------------------------------------------------
    def unfiltered_sample(self, size: int, maximum=None) -> np.ndarray:
        """Generate *size* combined points.

        ``maximum`` is forwarded to ``phsp2`` (majorant column for accept-reject),
        consistently with the TF original.
        """
        s1 = self.phsp1.unfiltered_sample(size)          # (size, d1)
        s2 = self.phsp2.unfiltered_sample(size, maximum) # (size, d2[+1])
        if s1.ndim == 1:
            s1 = s1[:, np.newaxis]
        if s2.ndim == 1:
            s2 = s2[:, np.newaxis]
        return np.concatenate([s1, s2], axis=1)

    def uniform_sample(self, size: int, maximum=None) -> np.ndarray:
        return self.filter(self.unfiltered_sample(size, maximum))