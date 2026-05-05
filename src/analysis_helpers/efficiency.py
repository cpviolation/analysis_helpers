import numpy as np


def get_sumw2(data, weights, bin_edges):
    """Calculates the sum of the squared weights in each bin.

    Args:
        data (array): data array
        weights (array): array of weights
        bin_edges (array): bin edges

    Returns:
        array: array with the sum of the squared weights in each bin
    """    
    hsumw2, _ = np.histogram(data, bins=bin_edges, range=range, weights=weights**2)
    return hsumw2


def weighted_quantile(values, weights, quantiles):
    """Calculates weighted quantile

    Args:
        values (array): Array of values to compute the quantiles for.
        weights (array): Array of weights corresponding to the values.
        quantiles (array): Array of quantiles to compute (between 0 and 1).

    Returns:
        array: Array of computed quantiles.
    """
    idx_sorted = np.argsort(values)
    sorted_values = values[idx_sorted]
    sorted_weights = weights[idx_sorted]

    total_weight = np.sum(sorted_weights)
    if total_weight <= 0:
        min_val = np.min(values) if len(values) > 0 else 0
        return np.full_like(quantiles, min_val, dtype=np.float64)

    cum_weights = np.cumsum(sorted_weights)
    normalized_cum_weights = (cum_weights - 0.5 * sorted_weights) / total_weight
    
    return np.interp(quantiles, normalized_cum_weights, sorted_values)


def weighted_std(values, weights):
    """Calculates the weighted standard deviation

    Args:
        values (array): Array of values to compute the standard deviation for.
        weights (array): Array of weights corresponding to the values.

    Raises:
        ValueError: If the sum of weights is not positive.

    Returns:
        float: Weighted standard deviation.
    """
    if np.sum(weights) <= 0:
        raise ValueError("Sum of weights must be positive to compute weighted standard deviation.")
    average = np.average(values, weights=weights)
    variance = np.average((values - average)**2, weights=weights)
    return np.sqrt(variance)



def logq(q, x):
    """q-logarithm function.

    Args:
        q (float): q parameter
        x (float): function argument

    Returns:
        float: result of the q-logarithm function
    """    
    return np.where(x >= 0, np.log(x) if q == 1 else (np.power(x, 1-q)-1)/(1-q), 
                    np.nan)


def variance_correction(n):
    """Variance correction function from Eq. 14 of 
    "Bias, variance, and confidence intervals for efficiency estimators in particle physics experiments", Dembinski et Schmelling (2022) - `arXiv:2110.00294<http://arxiv.org/abs/2110.00294>`_.

    Args:
        n (float): effective number of events

    Returns:
        float: function to correct the variance
    """    
    z_n = 1. / (1. + np.exp( -(logq(0.82, n)-logq(0.82, 2.92))/0.18 ) )
    f_n = (1 - z_n) * (n - n**2 / 4)
    f_n += z_n * (2*n + n**2 + n**3 + 6) / n**3
    return f_n


def wilson_interval(p, n, z=1.):
    """Wilson interval for a binomial distribution as described in Eq. 42 of
    "Bias, variance, and confidence intervals for efficiency estimators in particle physics experiments", Dembinski et Schmelling (2022) - `arXiv:2110.00294<http://arxiv.org/abs/2110.00294>`_.

    Args:
        p (float): probability of the binomial distribution
        n (float): effective number of events
        z (float, optional): size of the interval in standard deviations of a Normal. Defaults to 1..

    Returns:
        tuple: lower and upper bounds of the interval
    """
    f = variance_correction(n)
    b = z / n * np.sqrt( p * (1 - p) * n * f + z**2 / 4 * f**2 )
    p_i = p + z**2 / (2 * n) * f
    den = 1. / (1 + z**2 / n * f)
    return (p_i - b) * den, (p_i + b) * den


def binomial_error(p, n):
    """Expression of the binomial error

    Args:
        p (float): probability of the binomial distribution
        n (float): number of events

    Returns:
        float: binomial error
    """
    return np.sqrt(p * (1 - p) / n)


def get_efficiency(passed, total, bin_edges,
                   weights_passed=None, weights_total=None):
    """Efficiency calculation for a given set of passed and total events.

    Args:
        passed (array): events that passed the selection
        total (array): events that are in the total sample
        bin_edges (array): bin edges
        weights_passed (array, optional): weights of the events that passed the selection. Defaults to None.
        weights_total (array, optional): weights of the events that are in the total sample. Defaults to None.

    Returns:
        tuple: array with the efficiency and array with the lower and upper bounds of the efficiency
    """    
    ph, _ = np.histogram(passed, bins=bin_edges, weights=weights_passed)
    th, _ = np.histogram(total, bins=bin_edges, weights=weights_total)
    th_w2 = None
    if weights_total is not None:
        th_w2 = get_sumw2(total, weights_total, bin_edges)
    return get_efficiency_array(ph, th, th_w2)


def get_efficiency_array(passed, total, sumw2_total=None):
    """Efficiency calculation for a given set of passed and total events in a 1d array format.

    Args:
        passed (array): passed events
        total (array): total events
        sumw2_total (array, optional): the sum of squared weights for each bin. Defaults to None.

    Returns:
        (array, array): efficiency and confidence level
    """
    eff = np.divide(passed, total)
    if sumw2_total is not None:
        eff_cl = np.array(wilson_interval(eff, total**2 / sumw2_total))
    else:
        eff_err = np.array(binomial_error(eff, total))
        eff_cl = np.array([eff - eff_err[0], eff + eff_err[1]])
    # add protection against efficiency > 1 due to mostly events with negative weights being removed
    eff = np.where(eff > 1, 1, eff)
    eff = np.where(eff < 0, 0, eff)
    eff_cl[1] = np.where(eff == 1, 1, eff_cl[1])
    eff_cl[0] = np.where(eff == 0, 0, eff_cl[0])
    return eff, eff_cl


