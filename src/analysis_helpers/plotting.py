import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import mplhep
import math
from kinematics import estar2, estar3
from efficiency import get_efficiency, get_efficiency_array
from asymmetry import get_asymmetry_array
from matplotlib.colors import LogNorm
# mplhep.style.use("LHCb2")


def plot_hist(data, yerr=False, name=None, unit=None, ax=None, **kwargs):
    """Plot a histogram from an array

    Args:
        data (array): an array with the data to plot
        yerr (bool): if True, plot the y error bars
        name (str, optional): the variable name. Defaults to None.
        unit (str, optional): the unit name. Defaults to None.
        ax (axis, optional): matplotlib.axes object. Defaults to None.

    Returns:
        fig, ax: matplotlib figure and axis
    """
    # awkward to numpy if needed
    data_np_hist = type(data) is tuple and len(data) == 2
    if data_np_hist: # np.histogram
        dataA = data[0]
        yerr = np.sqrt(dataA) if yerr else None
        # needs to add sumw2
    elif data is ak.highlevel.Array:
        dataA = data.to_numpy()
    else:
        dataA = data
    # create figure if not existing
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    # set default values
    if 'bins' not in kwargs or data_np_hist:
        kwargs['bins'] = len(data[1])-1 if data_np_hist else 100 
    if 'range' not in kwargs or data_np_hist:
        kwargs['range'] = (data[1][0],data[1][-1]) if data_np_hist else (min(data), max(data))
    if 'histtype' not in kwargs:
        kwargs['histtype'] = 'step'
    if 'color' not in kwargs:
        kwargs['color'] = 'black'
    wei = kwargs['weights'] if 'weights' in kwargs else None
    den = kwargs['density'] if 'density' in kwargs else False
    # draw histogram
    hist = np.histogram(dataA, bins=kwargs['bins'], range=kwargs['range'], weights=wei, density=den) if not data_np_hist else data
    mplhep.histplot(
        hist,
        yerr=yerr,
        color=kwargs['color'],
        histtype=kwargs['histtype'],
        alpha=kwargs.get('alpha', None),
        label=kwargs.get('label', None),
        ax=ax,
    )
    # add labels
    bin_size = (kwargs['range'][1] - kwargs['range'][0]) / kwargs['bins']
    if unit is not None:
        ax.set_xlabel(f'{name} [{unit}]')
        ax.set_ylabel(f'Events / ({bin_size} {unit})')
    else:
        ax.set_xlabel(f'{name}')
        ax.set_ylabel(f'Events / ({bin_size})')
    # set x axis limits
    ax.set_xlim(kwargs['range'][0],kwargs['range'][1])
    return ax if fig is None else fig, ax


def plot_hists(data_arrays, yerrs=False, name=None, unit=None, ax=None, **kwargs):
    """Plot two histograms on the same axis using mplhep.histplot().

    Args:
        data_arrays (array<array>): Array with the data arrays for the each histogram.
        yerrs (list<bool>): for each data set if True, plot y error bars for the relevant histogram.
        name (str, optional): Name of the variable for the histograms. Defaults to None.
        unit (str, optional): Unit name for the histograms. Defaults to None.
        ax (axis, optional): Matplotlib axes object. Defaults to None.

    Returns:
        fig, ax: Matplotlib figure and axis.
    """
    dataA = [data.to_numpy() if data is ak.highlevel.Array else data for data in data_arrays]
    # Create figure and axes if not provided
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    # Set default values for kwargs
    if 'bins' not in kwargs:
        kwargs['bins'] = 100
    if 'range' not in kwargs:
        min_val = min([min(data) for data in data_arrays])
        max_val = max([max(data) for data in data_arrays])
        kwargs['range'] = (min_val, max_val)
    if 'histtype' not in kwargs:
        kwargs['histtype'] = 'step'
    if 'color' not in kwargs:
        kwargs['color'] = None
    if 'alpha' not in kwargs:
        kwargs['alpha'] = None
    weights = kwargs['weights'] if 'weights' in kwargs else None
    legends = kwargs['legend'] if 'legend' in kwargs else None
    den = kwargs['density'] if 'density' in kwargs else False
    # Compute histograms using np.histogram
    n_arrays = len(dataA)
    hists = [np.histogram(dataA[i], bins=kwargs['bins'], range=kwargs['range'], 
                          weights=None if weights is None else weights[i], 
                          density=den) for i in range(n_arrays)]

    # Plot the histograms
    for i in range(n_arrays):
        mplhep.histplot(hists[i], yerr=yerrs, histtype=kwargs['histtype'], alpha=kwargs['alpha'],
                        #color=kwargs['color'],
                          label=legends[i] if legends is not None else None, ax=ax)

    # Add labels
    bin_size = round((kwargs['range'][1] - kwargs['range'][0]) / kwargs['bins'], 3)
    #print(kwargs['range'][1], kwargs['range'][0], kwargs['bins'], bin_size)
    if unit is not None:
        ax.set_xlabel(f'{name} [{unit}]')
        ax.set_ylabel(f'Events / ({bin_size} {unit})')
    else:
        ax.set_xlabel(f'{name}')
        ax.set_ylabel(f'Events / ({bin_size})')

    # Set x axis limits
    ax.set_xlim(kwargs['range'][0], kwargs['range'][1])

    # Add legend
    if legends is not None:
        ax.legend()

    return ax if fig is None else fig, ax


def plot_double_hist(data1, data2, yerr1=False, yerr2=False, name=None, unit=None, ax=None, **kwargs):
    """Plot two histograms on the same axis using mplhep.histplot().

    Args:
        data1 (array): Array with the data for the first histogram.
        data2 (array): Array with the data for the second histogram.
        yerr1 (bool): If True, plot y error bars for the first histogram.
        yerr2 (bool): If True, plot y error bars for the second histogram.
        name1 (str, optional): Name of the variable for the first histogram. Defaults to None.
        name2 (str, optional): Name of the variable for the second histogram. Defaults to None.
        unit1 (str, optional): Unit name for the first histogram. Defaults to None.
        unit2 (str, optional): Unit name for the second histogram. Defaults to None.
        ax (axis, optional): Matplotlib axes object. Defaults to None.

    Returns:
        fig, ax: Matplotlib figure and axis.
    """

    dataA = data1.to_numpy() if data1 is ak.highlevel.Array else data1
    dataB = data2.to_numpy() if data2 is ak.highlevel.Array else data2
    # Create figure and axes if not provided
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    # Set default values for kwargs
    if 'bins' not in kwargs:
        kwargs['bins'] = 100
    if 'range' not in kwargs:
        min_val = min(min(data1), min(data2))
        max_val = max(max(data1), max(data2))
        kwargs['range'] = (min_val, max_val)
    if 'histtype' not in kwargs:
        kwargs['histtype'] = 'step'
    if 'color' not in kwargs:
        kwargs['color'] = 'black'
    wei1 = kwargs['weights1'] if 'weights1' in kwargs else None
    wei2 = kwargs['weights2'] if 'weights2' in kwargs else None
    leg1 = kwargs['legend1'] if 'legend1' in kwargs else None
    leg2 = kwargs['legend2'] if 'legend2' in kwargs else None
    den = kwargs['density'] if 'density' in kwargs else False
    # Compute histograms using np.histogram
    hist1 = np.histogram(dataA, bins=kwargs['bins'], range=kwargs['range'], weights=wei1, density=den)
    hist2 = np.histogram(dataB, bins=kwargs['bins'], range=kwargs['range'], weights=wei2, density=den)

    # Plot the first histogram
    mplhep.histplot(hist1, yerr=yerr1, histtype=kwargs['histtype'], color=kwargs['color'], label=leg1)

    # Plot the second histogram
    mplhep.histplot(hist2, yerr=yerr2, histtype=kwargs['histtype'], color='red', label=leg2, alpha=0.5)

    # Add labels
    bin_size = (kwargs['range'][1] - kwargs['range'][0]) / kwargs['bins']
    if unit is not None:
        ax.set_xlabel(f'{name} [{unit}]')
        ax.set_ylabel(f'Events / ({bin_size} {unit})')
    else:
        ax.set_xlabel(f'{name}')
        ax.set_ylabel(f'Events / ({bin_size})')

    # Set x axis limits
    ax.set_xlim(kwargs['range'][0], kwargs['range'][1])

    # Add legend
    ax.legend()

    return ax if fig is None else fig, ax


def plot_significance_and_purity(metrics, values, sel_name=None, axs=None, **kwargs):
    """Plots the significance and purity of a selection

    Args:
        metrics (dict): a dictionary with significance and purity values
        values (array): array of the selections
        sel_name (str, optional): the name of the selection. Defaults to None.

    Returns:
        fig, axs, ax2: matplotlib figure and axes
    """
    fig = None
    if axs is None:   
        fig, axs = plt.subplots(2, 2, figsize=[12, 7], 
                               gridspec_kw={'height_ratios': [0.1, 1], 'hspace': 0.3, 'wspace': 0.4})
        # Create legend axis (top row, spanning both columns)
        ax_legend = plt.subplot2grid((2, 2), (0, 0), colspan=2, fig=fig)
        ax_legend.axis('off')  # Hide the legend axis
        ax_legend.set_xticks([])  # Remove x-axis ticks
        ax_legend.set_yticks([])  # Remove y-axis ticks
        # Use the bottom row for the actual plots
        axs[0, 0].set_visible(False)  # Hide the unused top-left subplot
        axs[0, 1].set_visible(False)  # Hide the unused top-left subplot
        axs = axs[1]  # Use only the bottom row axes

    
    # unpack the metrics
    significance = np.array([k['significance'] for k in metrics])
    purity = np.array([k['purity'] for k in metrics])
    significance_purity = np.array([k['significance_purity'] for k in metrics])
    
    # Define colors for the three series
    colors = ['blue', 'red', 'green']
    
    # plot with specific colors
    line_s = axs[1].errorbar(values, significance[:,0], significance[:,1],
                             linestyle='', marker='o', label='S', color=colors[0], **kwargs)
    line_sp = axs[0].errorbar(values, significance_purity[:,0], significance_purity[:,1],
                             linestyle='', marker='o', label='S*P', color=colors[2], **kwargs)
    axs[1].set_xlabel('Selection' if sel_name is None else sel_name)
    axs[1].set_ylabel('Significance')
    axs[0].set_ylabel('Significance*Purity')
    axs[0].set_xlabel('Selection' if sel_name is None else sel_name)
    
    # Create second y-axis and plot the purity data
    ax2 = axs[1].twinx()
    line_p = ax2.errorbar(values, purity[:,0], purity[:,1],
                          color=colors[1], linestyle='', marker='o',
                          label='P', **kwargs)
    ax2.set_ylabel('Purity', color=colors[1])
    
    # Create common legend in the legend axis
    if fig is not None:
        lines = [line_s, line_p, line_sp]
        labels = ['Significance (S)', 'Purity (P)', 'Significance*Purity (S*P)']
        ax_legend.legend(lines, labels, loc='center', ncol=3, 
                        frameon=True, fancybox=False, shadow=False, fontsize='large')
    
    if fig is None:
        return axs, ax2
    return fig, axs, ax2


def plot_vertical_lines(ax, xline, color='red', linestyle='--'):
    """Plot vertical lines on an axis at specified values

    Args:
        ax (matplotlib.axes): axis object where to plot the lines
        xline (list): values of the x axis where to plot the lines
        color (str, optional): color of the lines. Defaults to 'red'.
        linestyle (str, optional): style of the line. Defaults to '--'.
    """
    for i in xline:
        ax.axvline(i, color=color, linestyle=linestyle)
    return


def plot_hist2d(x, y, ax=None, 
                xlabel=None, ylabel=None,
                vmax=None,
                **kwargs):
    """Plot and decorate a 2D histogram

    Args:
        x (array): the x values
        y (array): the y values
        ax (axis, optional): the axis where to plot the histogram. Defaults to None.
        xlabel (str, optional): x axis label. Defaults to None.
        ylabel (str, optional): y axis label. Defaults to None.
        vmax (float, optional): maximum value for the color scale. Defaults to None.

    Returns:
        fig, ax: figure and axis
    """    
    # create figure if not existing
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    # set default values
    bins = kwargs.get('bins', 100)
    if 'range' not in kwargs:
        kwargs['range'] = ( (np.min(x), np.max(x)), (np.min(y), np.max(y)) )
    wei = kwargs.get('weights', None)
    den = kwargs.get('density', False)
    hist2d = ax.hist2d(x, y, cmin=1, weights=wei, bins=bins, range=kwargs['range'], density=den, vmin=0, vmax=vmax)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    cbar = fig.colorbar(hist2d[3], ax=ax, label='Entries')
    # Set color bar ticks based on vmax if provided, otherwise use data maximum
    max_val = vmax if vmax is not None else np.max(hist2d[0][~np.isnan(hist2d[0])])
    cbar.set_ticks(np.linspace(0, max_val, 6))  # Set the color bar ticks
    try:
        fig.tight_layout()
        return fig, ax
    except Exception as e:
        print(f"Warning: tight_layout() failed due to: {e}")
        return ax
    

def plot_dalitz(m1, m2, xlab='', ylab='', masses=[], boundary=False, 
                weights=None, log=False, logz=False, bininfo=False, 
                ax=None, **kwargs):
    # awkward to numpy if needed
    m1a = m1.to_numpy() if m1 is ak.highlevel.Array else m1
    m2a = m2.to_numpy() if m2 is ak.highlevel.Array else m2
    
    nan_indices = np.isnan(m1a) | np.isnan(m2a)
    
    m1a = m1a[~nan_indices]
    m2a = m2a[~nan_indices]
    weights = weights[~nan_indices] if weights is not None else None
    # plot range
    if boundary:
        # needs to draw 2 lines: 1 above the diagonal and the other one below with the phase space limits
        # x needs to run between m12min=m1+m2 and m12max=M-m3
        if len(masses) != 4: 
            raise ValueError('Not enough masses to calculate the boundaries')
        if masses[0] < masses[1]+masses[2]+masses[3]:
            raise ValueError('The decay is kinematically not valid: M<m1+m2+m3. '\
                             'Be reminded that masses=[M,m1,m2,m3]')
        m1min = (masses[1]+masses[2])*(masses[1]+masses[2])
        m1max = (masses[0]-masses[3])*(masses[0]-masses[3])
        xwid = (m1max - m1min)
        xmin = m1min - 0.05*xwid
        xmax = m1max + 0.05*xwid
        m2min = (masses[3]+masses[2])*(masses[3]+masses[2])
        m2max = (masses[0]-masses[1])*(masses[0]-masses[1])
        ywid = (m2max - m2min)
        ymin = m2min - 0.05*ywid
        ymax = m2max + 0.05*ywid
    else:
        xwid = (m1a.max()-m1a.min())
        xmin = m1a.min() - 0.05*xwid
        xmax = m1a.max() + 0.05*xwid
        ywid = (m2a.max()-m2a.min())
        ymin = m2a.min() - 0.05*ywid
        ymax = m2a.max() + 0.05*ywid
    # wei = kwargs['weights'] if 'weights' in kwargs else None
    # den = kwargs['density'] if 'density' in kwargs else False
    # plot
    # create figure if not existing
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    #fig, ax = plt.subplots()
    # kwargs['norm'] = mpl.colors.Normalize(vmin=0)
    norm = LogNorm() if logz else mpl.colors.Normalize(vmin=0,vmax=kwargs.get('vmax', None))
    # Remove vmax from kwargs to avoid passing it twice
    kwargs_cleaned = {k: v for k, v in kwargs.items() if k != 'vmax'}
    h = ax.hist2d(m1a, m2a, cmin=1, range=[[xmin, xmax], [ymin, ymax]], weights=weights, norm=norm, **kwargs_cleaned)
    xedges = h[1]
    yedges = h[2]
    histogram = h[0]
    fig.colorbar(h[3], ax=ax, extend='max')
    if log:
        ax.set_yscale('log')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if boundary:
        # create an array with the X values:
        X = np.linspace(m1min, m1max, 10000)
        sqrtX = np.sqrt(X)
        # now calculate Ymin and Ymax
        est2, est3 = estar2(sqrtX, masses), estar3(sqrtX, masses)
        Ymin = np.power(est2+est3,2) - np.power(np.sqrt(est2*est2-masses[2]*masses[2])+ np.sqrt(est3*est3-masses[3]*masses[3]),2)
        Ymax = np.power(est2+est3,2) - np.power(np.sqrt(est2*est2-masses[2]*masses[2])- np.sqrt(est3*est3-masses[3]*masses[3]),2)
        ax.plot(X, Ymin, color='r')
        ax.plot(X, Ymax, color='r')
    if bininfo:
        return fig, ax, xedges, yedges, histogram
    else:
        return fig, ax


def plot_removed(data, selection, selection_label=None, 
                 yerr=False, name=None, unit=None, ax=None,
                 **kwargs):
    """Plots the histogram of the data and the data that has been removed by the selection

    Args:
        data (array): numpy array with the data
        selection (array): numpy array with the selection
        selection_label (str, optional): the name of the selection for the legend. Defaults to None.

    Returns:
        fig, ax: matplotlib figure and axis
    """
    label = kwargs.get('label', None)
    if label is None:
        label = 'data'
    else: 
        kwargs.pop('label')
    wei = None
    if 'weights' in kwargs:
        wei = kwargs['weights']
        kwargs.pop('weights')
    fig, ax = plot_hist(data, yerr=yerr, name=name, unit=unit, label=label, 
                        weights=wei, **kwargs)
    sel_label = selection_label if selection_label is not None else '(removed)'
    ax.hist(data[~selection], label=label+' '+sel_label, weights=wei[~selection], **kwargs)
    ax.legend()
    return fig, ax


def plot_signal_and_backgrond_distributions(data, variable, metric_variable,
                                            signal_range, sidebands_ranges, ax=None,
                                            **kwargs):
    """Plots signal and background distributions give the signal and sidebands 
    ranges

    Args:
        data (array): the numpy array with the data
        variable (str): the name of the variable to plot
        metric_variable (str): the name of the variable to cut on
        signal_range (list): a list with the signal range
        sidebands_ranges (array): a 2x2 list with the sidebands ranges `[sb1,sb2]`

    Returns:
        fig, ax: matplotlib figure and axis
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    signal = (signal_range[0] < data[metric_variable]) & \
             (data[metric_variable] < signal_range[1])
    sig_size = signal_range[1]-signal_range[0]
    side1  = (sidebands_ranges[0][0] < data[metric_variable]) & \
             (data[metric_variable] < sidebands_ranges[0][1])
    sb1_size= sidebands_ranges[0][1] - sidebands_ranges[0][0]
    side2  = (sidebands_ranges[1][0] < data[metric_variable]) & \
             (data[metric_variable] < sidebands_ranges[1][1])
    sb2_size= sidebands_ranges[1][1] - sidebands_ranges[1][0]
    sb_wei = sig_size/(sb1_size+sb2_size)
    ax.hist(np.concatenate((data[variable][signal],
                            data[variable][side1],
                            data[variable][side2])),
            weights=np.concatenate((np.ones_like(data[variable][signal]), 
                                    -np.ones_like(data[variable][side1])*sb_wei, 
                                    -np.ones_like(data[variable][side2])*sb_wei)),
            label='Signal', alpha=0.5, density=True, **kwargs)
    ax.hist(np.concatenate((data[variable][side1],
                            data[variable][side2])),
            weights=np.concatenate((np.ones_like(data[variable][side1])/sb1_size, 
                                    np.ones_like(data[variable][side2])/sb2_size)),
            label='Background', alpha=0.5, density=True, **kwargs)
    ax.legend()
    if fig is None:
        return ax
    return fig, ax


def adaptive_bin_edges(a, **kwargs):
    npa = a.to_numpy() if type(a) == ak.highlevel.Array else a
    xmin = np.min(npa)
    xmax = np.max(npa)    
    if 'range' in kwargs:
        xmin, xmax = kwargs['range']
    bin_edges = [xmin]
    x_bins = kwargs.get('bins', 10)
    for i in range(x_bins - 1):
        mask = np.logical_and(xmin <= npa, npa <= xmax)
        bin_edges.append(np.quantile(npa[mask], (i + 1) / x_bins))
    bin_edges.append(xmax)
    return bin_edges


def plot_efficiency(passed, total, 
                    name=None, unit=None, 
                    ax=None, **kwargs):
    """Plots the efficiency distribution of two datasets

    Args:
        passed (array): array with the events that passed the selection
        total (array): array with the total events in the sample
        name (str, optional): the variable name. Defaults to None.
        unit (str, optional): the unit name. Defaults to None.
        ax (axis, optional): axis of an already existing figure. Defaults to None.

    Returns:
        fig, ax: figure and axis
    """
    rng = kwargs.get('range', None)
    weights = kwargs.get('weights', None)
    wpassed = weights[0] if weights is not None else None
    wtotal = weights[1] if weights is not None else None
    if 'weights' in kwargs:
        kwargs.pop('weights')
    if rng is None:
        xmin = np.min( [np.min(passed), np.min(total)] )
        xmax = np.max( [np.max(passed), np.max(total)] )
        rng = (xmin, xmax)
    bin_edges = np.linspace(rng[0], rng[1], kwargs.get('bins', 10))
    eff, eff_cl = get_efficiency(passed, total, bin_edges,
                                 weights_passed=wpassed, weights_total=wtotal)
    yerr = [eff - eff_cl[0], eff_cl[1] - eff]
    x = []
    for i in range(len(bin_edges)-1):
        x += [ak.mean(total[np.multiply(bin_edges[i] < total,
                                        total < bin_edges[i+1])])]
    xerr_lower = [x[i] - bin_edges[i] for i in range(len(x))]
    xerr_higher = [bin_edges[i+1]-x[i] for i in range(len(x))]
    for k in ['range', 'bins']:
        if k in kwargs.keys():
            kwargs.pop(k)
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    ax.errorbar(x, eff, yerr=yerr, xerr=[xerr_lower, xerr_higher],
                linestyle='', marker='.', **kwargs)
    ax.set_ylabel('Efficiency')
    ax.set_ylim(0, 1.1)
    ax.set_xlabel(f'{name}'+(' [{unit}]' if unit is not None else ''))
    ax.set_xlim(rng[0], rng[1])
    ax.plot( ax.get_xlim(), [1,1], 'r--')
    return fig, ax if ax is None else ax


def plot_efficiency2d(passed, total, 
                    xname=None, xunit=None, 
                    yname=None, yunit=None, 
                    ax=None, **kwargs):
    """Plots the efficiency distribution of two datasets

    Args:
        passed (array): array with the events that passed the selection
        total (array): array with the total events in the sample
        name (str, optional): the variable name. Defaults to None.
        unit (str, optional): the unit name. Defaults to None.
        ax (axis, optional): axis of an already existing figure. Defaults to None.

    Returns:
        fig, ax: figure and axis
    """
    # extract weights
    weights = kwargs.get('weights', None)
    wpassed = weights[0] if weights is not None else None
    wtotal = weights[1] if weights is not None else None
    if 'weights' in kwargs:
        kwargs.pop('weights')
    # define binning scheme
    rng = kwargs.get('range', None)
    if rng is None:
        xmin = np.min( [np.min(passed[0]), np.min(total[0])] )
        xmax = np.max( [np.max(passed[0]), np.max(total[0])] )
        ymin = np.min( [np.min(passed[1]), np.min(total[1])] )
        ymax = np.max( [np.max(passed[1]), np.max(total[1])] )
        rng = ((xmin, xmax), (ymin, ymax))
    bins = kwargs.get('bins', 10)
    try: 
        xbins = bins[0]
        ybins = bins[1]
    except: 
        xbins = bins
        ybins = bins
    # calculate histograms
    xbin_edges = np.linspace(rng[0][0], rng[0][1], xbins+1)
    ybin_edges = np.linspace(rng[1][0], rng[1][1], ybins+1)
    hpassed = np.histogram2d(passed[0], passed[1], bins=(xbins,ybins), range=rng, 
                             weights=wpassed)
    htotal = np.histogram2d(total[0], total[1], bins=(xbins,ybins), range=rng, 
                            weights=wtotal)
    hsumw2_total = np.histogram2d(total[0], total[1], bins=(xbins,ybins), range=rng,
                                  weights=(wtotal**2 if wtotal is not None else None) )
    # calculate efficiency
    eff, eff_cl = get_efficiency_array(np.matrix.flatten(hpassed[0]), 
                                       np.matrix.flatten(htotal[0]), 
                                       np.matrix.flatten(hsumw2_total[0]))
    # yerr = [eff - eff_cl[0], eff_cl[1] - eff]
    # plot efficiency
    # build efficiency matrix
    eff_matrix = np.transpose(np.reshape(eff, (xbins,ybins)))
    # yerr_matrix = [np.transpose(np.reshape(eff - eff_cl[0], (xbins,ybins))), 
    #                np.transpose(np.reshape(eff_cl[1] - eff, (xbins,ybins)))]
    # plot
    for k in ['range', 'bins']:
        if k in kwargs.keys():
            kwargs.pop(k)
    fig = None
    if ax is None:
        fig, ax = plot_2darray(eff_matrix, xbins=xbin_edges, ybins=ybin_edges, 
                               zrange=(0,1), colorbar_label='Efficiency')
    else:
        plot_2darray(eff_matrix, xbins=xbin_edges, ybins=ybin_edges, 
                     zrange=(0,1), colorbar_label='Efficiency', ax=ax)
    ax.set_xlabel(f'{xname}'+(f' [{xunit}]' if xunit is not None else ''))
    ax.set_ylabel(f'{yname}'+(f' [{yunit}]' if yunit is not None else ''))
    return fig, ax if ax is None else ax


def plot_efficiency_with_distribution(passed, total,
                                      name=None, unit=None,
                                      ax=None, **kwargs):
    """Plots the efficiency distribution of two datasets with the total 
    distribution overlaid

    Args:
        passed (array): array with the events that passed the selection
        total (array): array with the total events in the sample
        name (str, optional): the variable name. Defaults to None.
        unit (str, optional): the unit name. Defaults to None.
        ax (axis, optional): axis of an already existing figure. Defaults to None.

    Returns:
        fig, axs: figure and axes
    """
    kWeights = 'weights' in kwargs
    wpassed = kwargs['weights'][0] if kWeights else None
    wtotal  = kwargs['weights'][1] if kWeights else None
    if 'weights' in kwargs:
        kwargs.pop('weights')
    if ax is None:
        fig, ax = plot_hist(total, weights=wtotal,
                            name=name, unit=unit,
                            color='grey', alpha=0.5, histtype='fill',
                            **kwargs)
    else:
        _ = plot_hist(total, weights=wtotal,
                      name=name, unit=unit, ax=ax,
                      color='grey', alpha=0.5, histtype='fill',
                      **kwargs)
    ax2 = ax.twinx()
    _ = plot_efficiency(passed, total,
                        weights=(wpassed, wtotal) if kWeights else None,
                        label='efficiency', ax=ax2,
                        **kwargs)
    # legend
    handles1, labels1 =  ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    ax2.legend(handles, labels, loc='lower right')
    ax2.legend()
    try:
        fig.tight_layout()
        return fig, (ax, ax2)
    except Exception as e:
        print(f"Warning: tight_layout() failed due to: {e}")
        return (ax, ax2)


def plot_asymmetry2d(arr1, arr2, 
                     xname=None, xunit=None, 
                     yname=None, yunit=None, 
                     ax=None, **kwargs):
    
    """Plots the asymmetry between two datasets

    Args:
        arr1 (array): array with the events from the first sample
        arr2 (array): array with the events from the second sample
        xname (str, optional): the variable on the x axis name. Defaults to None.
        xunit (str, optional): the variable on the x axis unit name. Defaults to None.
        yname (str, optional): the variable on the y axis name. Defaults to None.
        yunit (str, optional): the variable on the y axis unit name. Defaults to None.
        ax (axis, optional): axis of an already existing figure. Defaults to None.

    Returns:
        fig, ax: figure and axis
    """
    # extract weights
    weights = kwargs.get('weights', None)
    warr1 = weights[0] if weights is not None else None
    warr2 = weights[1] if weights is not None else None
    if 'weights' in kwargs:
        kwargs.pop('weights')
    # define binning scheme
    rng = kwargs.get('range', None)
    if rng is None:
        xmin = np.min( [np.min(arr1[0]), np.min(arr2[0])] )
        xmax = np.max( [np.max(arr1[0]), np.max(arr2[0])] )
        ymin = np.min( [np.min(arr1[1]), np.min(arr2[1])] )
        ymax = np.max( [np.max(arr1[1]), np.max(arr2[1])] )
        rng = ((xmin, xmax), (ymin, ymax))
    bins = kwargs.get('bins', 10)
    try: 
        xbins = bins[0]
        ybins = bins[1]
    except: 
        xbins = bins
        ybins = bins
    # calculate histograms
    xbin_edges = np.linspace(rng[0][0], rng[0][1], xbins+1)
    ybin_edges = np.linspace(rng[1][0], rng[1][1], ybins+1)
    harr1 = np.histogram2d(arr1[0], arr1[1], bins=(xbins,ybins), range=rng, 
                             weights=warr1)
    harr2 = np.histogram2d(arr2[0], arr2[1], bins=(xbins,ybins), range=rng, 
                            weights=warr2)
    # calculate asymmetry
    #asym, asym_unc = get_asymmetry_array(np.matrix.flatten(harr1[0]), 
    asym, _ = get_asymmetry_array(np.matrix.flatten(harr1[0]), 
                                      np.matrix.flatten(harr2[0]), 
                                      np.matrix.flatten(warr1[0]) if warr1 is not None else None, 
                                      np.matrix.flatten(warr2[0]) if warr2 is not None else None)
    #asym_err = [asym - asym_unc, asym + asym_unc]
    # plot asymmetry
    # build efficiency matrix
    asym_matrix = np.transpose(np.reshape(asym, (xbins,ybins)))
    # asym_err_matrix = [np.transpose(np.reshape(asym_err[0], (xbins,ybins))), 
    #                    np.transpose(np.reshape(asym_err[1], (xbins,ybins)))]
    # plot
    for k in ['range', 'bins']:
        if k in kwargs.keys():
            kwargs.pop(k)
    fig = None
    if ax is None:
        fig, ax = plot_2darray(asym_matrix, xbins=xbin_edges, ybins=ybin_edges, 
                               zrange=(-1,1), colorbar_label='Asymmetry')
    else:
        plot_2darray(asym_matrix, xbins=xbin_edges, ybins=ybin_edges, 
                     zrange=(0,1), colorbar_label='Asymmetry', ax=ax)
    ax.set_xlabel(f'{xname}'+(f' [{xunit}]' if xunit is not None else ''))
    ax.set_ylabel(f'{yname}'+(f' [{yunit}]' if yunit is not None else ''))
    return fig, ax if ax is None else ax


def create_subplots(num_plots, figsize=(10, 6)):
    """Creates a figure and axes for a given number of plots

    Args:
        num_plots (int): The total number of plots to create.
        figsize (tuple, optional): The size of the overall figure, specified as (width, height). Default is (10, 6).


    Returns:
        matplotlib.figure.Figure : The Matplotlib Figure object containing the subplots.
        list of matplotlib.axes._subplots.AxesSubplot : A flat list of Axes objects corresponding to the subplots. If the grid has more subplots than `num_plots`, unused axes are hidden.

    Example:

    .. code-block:: python

        >>> num_plots = 7
        >>> fig, axes = create_subplots(num_plots)
        >>> for i in range(num_plots):
        >>>     axes[i].plot([0, 1, 2], [i, i+1, i+2])
        >>>     axes[i].set_title(f"Plot {i+1}")
        >>> plt.tight_layout()
        >>> plt.show()

    """
    # Determine number of rows and columns
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)
    # Adjust font size dynamically
    base_font_size = 14.0  # Base font size for titles and labels
    font_scale = max(1, math.sqrt(num_plots) / 2)  # Scale font size with sqrt(num_plots)
    plt.rcParams.update({'font.size': base_font_size / font_scale})
    # Create the figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    # Flatten the axes array for easy iteration (even if it's 2D)
    axes = axes.flatten() if num_plots > 1 else [axes]
    # Turn off unused axes
    for ax in axes[num_plots:]:
        ax.set_visible(False)
    #plt.rcParams.update({'font.size': base_font_size})
    return fig, axes


def plot_2darray(data, xbins=None, ybins=None, zrange=None,
                 xlabel='', ylabel='', colorbar_label='', 
                 title='', ax=None, lines=False):
    """Plots a 2D array as a heatmap

    Args:
        data (array): the data to plot
        xbins (array, optional): binning on the x axis. Defaults to None.
        ybins (array, optional): binning on the y axis. Defaults to None.
        zrange (tuple, optional): z-axis range as (min,max). Defaults to None.
        colorbar_label (str, optional): the color bar label. Defaults to ''.
        ax (axis, optional): the axis where to make the plot. Defaults to None.

    Returns:
        fig, axs: figure and axes
    """    
    # Creates a coordinates grid
    xbins = np.linspace(0, data.shape[0], data.shape[0]+1) if xbins is None else xbins
    ybins = np.linspace(0, data.shape[1], data.shape[1]+1) if ybins is None else ybins
    x_grid, y_grid = np.meshgrid(
        0.5 * (xbins[:-1] + xbins[1:]),
        0.5 * (ybins[:-1] + ybins[1:]),
        indexing='ij'
    )
    # Plot
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    zrange = (np.min(data), np.max(data)) if zrange is None else zrange
    vmin, vmax = zrange  # Set the desired color range
    c = ax.pcolormesh(x_grid, y_grid, data.T,
                      shading='auto', cmap='viridis',
                      vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(c, ax=ax, label=colorbar_label)
    cbar.set_ticks(np.linspace(vmin, vmax, 6))  # Set the color bar ticks
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Add grid lines if specified
    if lines:
        for x in xbins:
            ax.axvline(x=x, color='k', linestyle='--', linewidth=1)
        for y in ybins:
            ax.axhline(y=y, color='k', linestyle='--', linewidth=1)
    try:
        fig.tight_layout()
        return fig, ax
    except Exception as e:
        print(f"Warning: tight_layout() failed due to: {e}")
        return ax


def configure_plot_layout(fig, method='tight_layout'):
    """Configure plot layout to prevent text overlapping.

    Args:
        fig (matplotlib.figure.Figure): The matplotlib figure.
        method (str): Layout method ('tight_layout', 'constrained', 'subplots_adjust').
    """
    if method == 'constrained':
        # Use constrained layout (newer matplotlib feature)
        fig.set_constrained_layout(True)
    elif method == 'subplots_adjust':
        # Manual adjustment of subplot parameters
        fig.subplots_adjust(
            left=0.1,    # left margin
            bottom=0.15, # bottom margin
            right=0.95,  # right margin
            top=0.9,     # top margin
            wspace=0.4,  # width spacing between subplots
            hspace=0.5   # height spacing between subplots
        )
    else:  # default: tight_layout
        plt.tight_layout(pad=1.5, h_pad=2.0, w_pad=1.5)
