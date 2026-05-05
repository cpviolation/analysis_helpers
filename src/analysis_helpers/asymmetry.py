import numpy as np

def get_asymmetry_array(arr1, arr2,
                        sumw2_arr1=None, sumw2_arr2=None,
                        correlation=0.):
    r"""Calculates the asymmetry between two arrays

    - $a = \frac{n_1- n_2}{n_1 + n_2}$
    - $\sigma_a^2 = 4\frac{n_2^2 \sigma_1^2 + n_1^2\sigma_2^2 - 2n_1 n_2 \rho \sigma_1\sigma_2}{ \left(n_1 + n_2\right)^4}$

    Args:
        arr1 (array): array with the first set of values
        arr2 (array): array with the second set of values
        sumw2_arr1 (array, optional): array with the squared sum of weights for first set of values. Defaults to None.
        sumw2_arr2 (array, optional): array with the squared sum of weights for second set of values. Defaults to None.
        correlation (float, optional): correlation between the two data sets. Defaults to 0.

    Returns:
        _type_: _description_
    """    
    # Asymmetry calculation
    den = np.where(arr1 + arr2, arr1 + arr2, 1)
    num =  arr1 - arr2
    asym = np.divide(num, den)
    # Uncertainty calculation
    arr1_err = np.sqrt(sumw2_arr1) if sumw2_arr1 is not None else np.sqrt(arr1)
    arr2_err = np.sqrt(sumw2_arr2) if sumw2_arr2 is not None else np.sqrt(arr2)
    err2_num = 4.*( arr2**2 * arr1_err**2 + arr1**2 * arr2_err**2 \
                   - 2*correlation*arr1*arr2*arr1_err*arr2_err )
    err2_den = den**4
    asym_err = np.sqrt( np.divide( err2_num, err2_den ) )
    return asym, asym_err
