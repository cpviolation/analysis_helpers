import numpy as np

def covariance_from_correlation(rho_percent, sigma):
    """
    Calculate the covariance matrix from the correlation coefficient and standard deviations.

    Args:
        rho_percent (float): The correlation coefficient in percentage (0-100).
        sigma (array-like): The standard deviations of the variables.

    Returns:
        numpy.ndarray: The covariance matrix.
    """
    rho = rho_percent / 100.0
    return rho * np.outer(sigma, sigma)


def unc_ratio(n1,n2,s1=None,s2=None):
    """
    Calculate the uncertainty of the ratio of two independent measurements.

    Args:
        n1 (float): The first measurement.
        n2 (float): The second measurement.
        s1 (float, optional): The uncertainty of the first measurement. If not provided, it will be calculated as sqrt(n1).
        s2 (float, optional): The uncertainty of the second measurement. If not provided, it will be calculated as sqrt(n2).

    Returns:
        float: The uncertainty of the ratio n1/n2.
    """
    if s1 is None:
        s1 = np.sqrt(n1)
    if s2 is None:
        s2 = np.sqrt(n2)
    unc = (n1 / n2) * np.sqrt((s1 / n1)**2 + (s2 / n2)**2)
    return unc