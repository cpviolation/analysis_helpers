import numpy as np


def significance(Nsb, Nb, sf=1., Nsb_err=None, Nb_err=None):
    """Calculate the significance of a signal given the number of signal and 
    background events and their uncertainties.

    Args:
        Nsb (float): the number of signal+background events
        Nb (float): the number of background events
        sf (float, optional): the normalisation factor for background events in the signal region. Defaults to 1.
        Nsb_err (float, optional): the uncertainty on the number of signal+background events. Defaults to None to assume Poisson statistics.
        Nb_err (float, optional): the uncertainty on the number of background events. Defaults to None to assume Poisson statistics.

    Returns:
        (float,float): the significance of the signal and its uncertainty
    """
    Ns = Nsb - Nb * sf
    significance = Ns/np.sqrt(Nsb)
    if Nsb_err is None: 
        Nsb_err = np.sqrt(Nsb)
    if Nb_err is None: 
        Nb_err = np.sqrt(Nb)
    significance_error = np.sqrt(0.25 * np.power(1.+Nb/Nsb*sf, 2) * Nsb_err * Nsb_err/Nsb + 
                                 sf * sf * Nb_err * Nb_err / Nsb )
    return (significance, significance_error)

def purity(Nsb, Nb, sf=1., Nsb_err=None, Nb_err=None):
    """Calculate the significance of a signal given the number of signal and 
    background events and their uncertainties.

    Args:
        Nsb (float): the number of signal+background events
        Nb (float): the number of background events
        sf (float, optional): the normalisation factor for background events in the signal region. Defaults to 1.
        Nsb_err (float, optional): the uncertainty on the number of signal+background events. Defaults to None to assume Poisson statistics.
        Nb_err (float, optional): the uncertainty on the number of background events. Defaults to None to assume Poisson statistics.

    Returns:
        (float,float): the purity of the signal and its uncertainty
    """
    if Nsb_err is None: 
        Nsb_err = np.sqrt(Nsb)
    if Nb_err is None: 
        Nb_err = np.sqrt(Nb)
    purity = 1.-sf*Nb/Nsb
    purity_error = sf / Nsb * np.sqrt( Nb*Nb/Nsb/Nsb*Nsb_err*Nsb_err + Nb_err*Nb_err )
    return (purity, purity_error)

def significance_purity(Nsb, Nb, sf=1., Nsb_err=None, Nb_err=None):
    """Calculate the significance of a signal given the number of signal and 
    background events and their uncertainties.

    Args:
        Nsb (float): the number of signal+background events
        Nb (float): the number of background events
        sf (float, optional): the normalisation factor for background events in the signal region. Defaults to 1.
        Nsb_err (float, optional): the uncertainty on the number of signal+background events. Defaults to None to assume Poisson statistics.
        Nb_err (float, optional): the uncertainty on the number of background events. Defaults to None to assume Poisson statistics.

    Returns:
        (float,float): the significance*purity of the signal and its uncertainty
    """
    if Nsb_err is None: 
        Nsb_err = np.sqrt(Nsb)
    if Nb_err is None:
        Nb_err = np.sqrt(Nb)
    sp = np.power(Nsb - Nb*sf,2)/np.power(Nsb,1.5)
    sp_err = np.sqrt( np.power(0.5 -1.5*sf*sf*Nb*Nb/Nsb/Nsb +sf*Nb/Nsb,2)*Nsb_err*Nsb_err/Nsb + 4.*sf*sf/Nsb*np.power( sf*Nb/Nsb -1,2)*Nb_err*Nb_err )
    return (sp, sp_err)

def calculate_significance_and_purity(data, signal_range, sidebands_ranges):
    """Given a data array, a signal range and two sidebands ranges, 
    calculates the significance of the signal.

    Args:
        data (ndarray): a data array
        signal_range (ndarray): a 2-element array with the signal range
        sidebands_ranges (ndarray): a 2x2 array with the sidebands ranges

    Returns:
        dict: a dictionary with the significance, purity and significance*purity of the signal and their uncertainty
    """
    signal = (signal_range[0] < data) & (data < signal_range[1])
    sig_size = signal_range[1]-signal_range[0]
    Nb, sb_size = 0, 0
    for sb_range in sidebands_ranges:
        sb_size = sb_range[1]-sb_range[0]
        Nb += np.sum( (sb_range[0] < data) & (data < sb_range[1]) )
    # side1 = np.multiply(sidebands_ranges[0][0] < data,
    #                     data < sidebands_ranges[0][1])
    # sb1_size = sidebands_ranges[0][1] - sidebands_ranges[0][0]
    # side2 = np.multiply(sidebands_ranges[1][0] < data,
    #                     data < sidebands_ranges[1][1])
    # sb2_size = sidebands_ranges[1][1] - sidebands_ranges[1][0]
    # Nb = np.sum(side1)+np.sum(side2)
    Nsb = np.sum(signal)
    #sf = sig_size / (sb1_size+sb2_size)
    sf = sig_size / sb_size
    sign = significance(Nsb, Nb, sf)
    puri = purity(Nsb, Nb, sf)
    sign_puri = significance_purity(Nsb, Nb, sf)
    return {'significance': sign, 
            'purity': puri, 
            'significance_purity': sign_puri}

def significance_and_purity_values(data,variable, metric_variable,
                                signal_range, sidebands_ranges, selections,
                                selection_type='>'):
    metrics_values = []
    for sel in selections:
        if selection_type == '>':
            selection_mask = data[variable] > sel
        elif selection_type == '<':
            selection_mask = data[variable] < sel
        else:
            raise ValueError('Invalid selection type')
        metrics_values.append(calculate_significance_and_purity(data[selection_mask][metric_variable],
                                                             signal_range, sidebands_ranges))
    return metrics_values

def calculate_significance_and_puritySW(data, signal_weights, background_weights):
    """Given arrays of data, signal and background weights,
    calculates the significance of the signal.

    Args:
        data (ndarray): a data array
        signal_weights (ndarray): a weights array for the signal
        background_weights (ndarray): a weights array for the background

    Returns:
        dict: a dictionary with the significance, purity and significance*purity of the signal and their uncertainty
    """
    Ns = signal_weights.sum()
    Nb = background_weights.sum()
    Nsb = Ns+Nb
    significance = significance(Nsb, Nb, 1.)
    purity = purity(Nsb, Nb, 1.)
    significance_purity = significance_purity(Nsb, Nb, 1.)
    return {'significance': significance, 
            'purity': purity, 
            'significance_purity': significance_purity}

def significance_and_purity_valuesSW(data, variable, metric_variable,
                                   signal_weights, background_weights, selections,
                                   selection_type='>'):
    metrics_values = []
    for sel in selections:
        if selection_type == '>':
            selection_mask = data[variable] > sel
        elif selection_type == '<':
            selection_mask = data[variable] < sel
        else:
            raise ValueError('Invalid selection type')
        metrics_values.append(calculate_significance_and_puritySW(data[selection_mask][metric_variable],
                                                               signal_weights[selection_mask],
                                                               background_weights[selection_mask]))
    return metrics_values
