#import awkward as ak
import numpy as np
import scipy.constants

def ip_all(a_x,a_y,a_z,n_x,n_y,n_z,p_x,p_y,p_z):
    r"""Function to calculate the impact parameter of a point $\vec{p}$ from a line 
    passing from a point $\vec{a}$ and having direction $\vec{n}$

    Args:
        a_x, a_y, a_z (float,float,float): coordinates (x,y,z) of a point on the line
        n_x, n_y, n_z (float,float,float): magnitudes (x,y,z) of the unit vector of the line direction
        p_x. p_y, p_z (float,float,float): coordinates (x,y,z) of the point
    """
    p_min_a_x = p_x-a_x
    p_min_a_y = p_y-a_y
    p_min_a_z = p_z-a_z
    #
    cross_p_x = p_min_a_y*n_z- n_y*p_min_a_z
    cross_p_y = p_min_a_z*n_x- n_z*p_min_a_x
    cross_p_z = p_min_a_x*n_y- n_x*p_min_a_y
    #
    num = np.sqrt(cross_p_x*cross_p_x + cross_p_y*cross_p_y + cross_p_z*cross_p_z)
    den = np.sqrt(n_x*n_x + n_y*n_y + n_z*n_z)
    return num/den

def doca(Ax,Ay,Az,Atx,Aty,
         Bx,By,Bz,Btx,Bty):
    r"""Calculate the Distance of Closest Approach (DOCA) between two lines in 3D space.

    Args:
        Ax (float): x-coordinate of a point on line A
        Ay (float): y-coordinate of a point on line A
        Az (float): z-coordinate of a point on line A
        Atx (float): x-component of the direction vector of line A
        Aty (float): y-component of the direction vector of line A
        Bx (float): x-coordinate of a point on line B
        By (float): y-coordinate of a point on line B
        Bz (float): z-coordinate of a point on line B
        Btx (float): x-component of the direction vector of line B
        Bty (float): y-component of the direction vector of line B

    Returns:
        float: the Distance of Closest Approach (DOCA) between the two lines
    """    
    secondAA =  Atx * Atx + Aty * Aty + 1.
    secondBB =  Btx * Btx + Bty * Bty + 1.
    secondAB = -Atx * Btx - Aty * Bty - 1.
    det = secondAA * secondBB - secondAB * secondAB
    #ret = -1.
    #if type(det) == ak.highlevel.Array: det = det.to_numpy()
    #if np.abs(det) > 0:
    secondinvAA =  secondBB / det
    secondinvBB =  secondAA / det
    secondinvAB = -secondAB / det
    firstA =  Atx * (Ax - Bx) + Aty * (Ay - By) + (Az - Bz)
    firstB = -Btx * (Ax - Bx) - Bty * (Ay - By) - (Az - Bz)
    muA = -(secondinvAA * firstA + secondinvAB * firstB)
    muB = -(secondinvBB * firstB + secondinvAB * firstA)
    dx = (Ax + muA * Atx) - (Bx + muB * Btx)
    dy = (Ay + muA * Aty) - (By + muB * Bty)
    dz = (Az + muA) - (Bz + muB)
    ret = np.sqrt(dx * dx + dy * dy + dz * dz)
    return ret

def dira(Px,Py,Pz,
         Vx,Vy,Vz,
         PVx,PVy,PVz):
    """Calculate the Directional Angle (DIRA) between a particle and a vertex.

    Args:
        Px (float): x-component of the particle's momentum
        Py (float): y-component of the particle's momentum
        Pz (float): z-component of the particle's momentum
        Vx (float): x-coordinate of the vertex
        Vy (float): y-coordinate of the vertex
        Vz (float): z-coordinate of the vertex
        PVx (float): x-coordinate of the primary vertex
        PVy (float): y-coordinate of the primary vertex
        PVz (float): z-coordinate of the primary vertex

    Returns:
        float: the Directional Angle (DIRA) between the particle and the vertex
    """    
    dx, dy, dz = Vx-PVx, Vy-PVy, Vz-PVz
    fd = np.sqrt(dx*dx+dy*dy+dz*dz)
    p = np.sqrt(Px*Px+Py*Py+Pz*Pz)
    dira = np.nan_to_num( (dx*Px + dy*Py + dz*Pz) / (p*fd), nan=0, posinf=np.inf, neginf=-np.inf)
    return dira

def opening_angle(P1x,P1y,P1z,
                  P2x,P2y,P2z):
    """Calculate the opening angle between two 3D vectors.

    Args:
        P1x (float): x-component of the first vector
        P1y (float): y-component of the first vector
        P1z (float): z-component of the first vector
        P2x (float): x-component of the second vector
        P2y (float): y-component of the second vector
        P2z (float): z-component of the second vector

    Returns:
        float: the opening angle between the two vectors
    """    
    P1_tx,P1_ty,P2_tx,P2_ty = P1x/P1z, P1y/P1z, P2x/P2z, P2y/P2z
    oa_norm = np.sqrt( (P1_tx*P1_tx + P1_ty*P1_ty + 1) * (P2_tx*P2_tx + P2_ty*P2_ty + 1) )
    oa_arg  = (P1_tx * P2_tx + P1_ty * P2_ty + 1.) / oa_norm
    opening_angle = np.nan_to_num( np.arccos(oa_arg), nan=0, posinf=np.inf, neginf=-np.inf)
    return opening_angle

def ctau(L,m,p):
    """Calculate the proper time (CTAU) of a particle.

    Args:
        L (float): length of the particle's trajectory in millimeters
        m (float): mass of the particle in MeV/c^2
        p (float): momentum of the particle in MeV/c per millimeter

    Returns:
        float: proper time (CTAU) of the particle in millimeters
    """    
    ct = m*L/p # mm
    return ct

def ctau_from_tau(tau):
    """Calculate the proper time (CTAU) from the lifetime (TAU) of a particle.

    Args:
        tau (float): lifetime of the particle in nanoseconds

    Returns:
        float: proper time (CTAU) of the particle in millimeters
    """    
    # from ns to mm
    return  tau * scipy.constants.c / 1e6

def distance(x0,y0,z0,x1,y1,z1):
    """Calculate the distance between two points in 3D space.

    Args:
        x0 (float): x-coordinate of the first point
        y0 (float): y-coordinate of the first point
        z0 (float): z-coordinate of the first point
        x1 (float): x-coordinate of the second point
        y1 (float): y-coordinate of the second point
        z1 (float): z-coordinate of the second point

    Returns:
        float: distance between the two points
    """    
    dl = np.sqrt( (x0-x1)*(x0-x1) + (y0-y1)*(y0-y1) + (z0-z1)*(z0-z1) )
    return dl

def mass(px,py,pz,pe):
    """Calculate the invariant mass (M) of a particle.

    Args:
        px (float): x-component of the particle's momentum
        py (float): y-component of the particle's momentum
        pz (float): z-component of the particle's momentum
        pe (float): energy of the particle

    Returns:
        float: invariant mass of the particle
    """    
    m = np.sqrt( pe*pe - px*px - py*py - pz*pz )
    return m

def momentum(px,py,pz):
    """Calculate the momentum (P) of a particle.

    Args:
        px (float): x-component of the particle's momentum
        py (float): y-component of the particle's momentum
        pz (float): z-component of the particle's momentum

    Returns:
        float: momentum of the particle
    """    
    p = np.sqrt(px*px + py*py + pz*pz)
    return p

def pt(px,py):
    """Calculate the transverse momentum (PT) of a particle.

    Args:
        px (float): x-component of the particle's momentum
        py (float): y-component of the particle's momentum

    Returns:
        float: transverse momentum of the particle
    """    
    pt = np.sqrt(px*px + py*py)
    return pt

def rapidity(pa,pz):
    """Calculate the rapidity momentum (T) of a particle.

    Args:
        pa (float): transverse momentum of the particle
        pz (float): longitudinal momentum of the particle

    Returns:
        float: rapidity momentum of the particle
    """    
    t = pa/pz
    return t

def pseudorapidity(px,py,pz):
    """Calculate the pseudorapidity (ETA) of a particle.

    Args:
        px (float): x-component of the particle's momentum
        py (float): y-component of the particle's momentum
        pz (float): z-component of the particle's momentum

    Returns:
        float: the pseudorapidity (ETA) of the particle
    """    
    eta = np.arctanh(pz/P(px,py,pz))
    return eta

def estar2(m12,masses):
    """Calculate the E* of a two-body system.

    Args:
        m12 (float): invariant mass of the first two particles
        masses (list): list of masses of the four particles

    Returns:
        float: the E* of the two-body system
    """
    Est2 = m12*m12 - masses[1]*masses[1] + masses[2]*masses[2]
    Est2/= 2*m12
    return Est2

def estar3(m12,masses):
    """Calculate the E* of a three-body system.

    Args:
        m12 (float): invariant mass of the first two particles
        masses (list): list of masses of the four particles

    Returns:
        float: the E* of the three-body system
    """
    Est3 = masses[0]*masses[0]- m12*m12 - masses[3]*masses[3]
    Est3/= 2*m12
    return Est3