import numpy as np
from .efficiency import weighted_std, weighted_quantile
from .utils import to_numpy, get_mask

def adaptive_bin_edges(a, bins=10, bins_range=None, weights=None):
    """Calculates bin edges from quantiles

    Args:
        a (array): array of data to bin.
        bins (int): number of bins. Defaults to 10.
        range (tuple, optional): range of the data to consider. Defaults to None.
        weights (array, optional): weights for the data. Defaults to None.

    Returns:
        array: array of bin edges.
    """    
    npa = to_numpy(a)
    xmin = np.min(npa)
    xmax = np.max(npa)    
    if bins_range is not None:
        xmin, xmax = bins_range
    bin_edges = [xmin]
    x_bins = bins
    if weights is not None: weights = to_numpy(weights)
    mask = np.logical_and(xmin <= npa, npa <= xmax)
    for i in range(x_bins - 1):
        if weights is not None:
            bin_edges.append( weighted_quantile(
                npa[mask], weights[mask], (i + 1) / x_bins) )
        else:
            bin_edges.append(np.quantile(npa[mask], (i + 1) / x_bins))
    bin_edges.append(xmax)
    return bin_edges


def adaptive_bin_edges_2d(a, bins=(10,10), bins_range=None, weights=None, independent=False):
    """Adaptive bin edges in 2d

    Args:
        a (array): 2d-array of data to bin.
        bins (tuple, optional): Number of bins for each dimension. Defaults to (10,10).
        bins_range (tuple, optional): Range for each dimension. Defaults to None.
        weights (array, optional): Weights for the data. Defaults to None.
        independent (bool, optional): independent binning for each axis. Defaults to False.

    Raises:
        ValueError: Input data must be a 2D array with shape (N, 2).

    Returns:
        dict: Dictionary with bin edges for each dimension.
    """    
    npa = to_numpy(a)
    if npa.ndim != 2 or npa.shape[1] != 2:
        raise ValueError("Input data must be a 2D array with shape (N, 2).")
    bin_edges = {}
    bin_edges_x = adaptive_bin_edges(npa[:,0], bins=bins[0], 
                                     bins_range=bins_range[0] if bins_range is not None else None, weights=weights)
    bin_edges[0] = bin_edges_x
    if independent:
        bin_edges[1] = adaptive_bin_edges(npa[:,1], bins=bins[1], 
                                          bins_range=bins_range[1] if bins_range is not None else None, weights=weights)
    else:                                      
        bin_edges[1] = {}
        for i in range(len(bin_edges_x)-1):
            mask_x = (npa[:,0] >= bin_edges_x[i]) & (npa[:,0] < bin_edges_x[i+1])
            if weights is not None:
                bin_edges_y = adaptive_bin_edges(
                    npa[mask_x,1], bins=bins[1],
                    bins_range=bins_range[1] if bins_range is not None else None,
                    weights=weights[mask_x])
            else:
                bin_edges_y = adaptive_bin_edges(
                    npa[mask_x,1], bins=bins[1],
                    bins_range=bins_range[1] if bins_range is not None else None)
            bin_edges[1][i] = bin_edges_y
    return bin_edges


def adaptive_bin_edges_3d(a, bins=None, bins_range=None, weights=None, 
                        target_events_per_3d_bin=None, 
                        strategy='proportional',
                        independent=False,
                        verbose=False):
    """Creates 3D adaptive binning recursively with a target of events per cell.
    
    This function recursively bins each dimension:
    1. Bins X globally
    2. For each X bin, bins Y within that X slice
    3. For each (X,Y) bin, bins Z within that slice
    
    This creates truly adaptive binning where each dimension's edges depend on
    the data distribution within the previous dimensions' bins.

    Args:
        a (array): The data to bin.
        bins (tuple, optional): Number of bins for each dimension. Defaults to None.
        bins_range (tuple, optional): Range for each dimension. Defaults to None.
        weights (np.ndarray, optional): The event weights.
        target_events_per_3d_bin (int): The target number of events per 3D bin.
        strategy (str, optional): The binning strategy ('equal' or 'proportional'). Defaults to 'proportional'.
        independent (bool, optional): independent binning for each axis. Defaults to False.
        verbose (bool, optional): Enable verbose output. Defaults to False.

    Raises:
        ValueError: If the input data is not 1D.
        ValueError: If the input data arrays have different lengths.
        ValueError: If the target_events_per_3d_bin is not positive.

    Returns:
        dict: Dictionary with nested structure:
            - bin_edges[0]: X edges (1D array)
            - bin_edges[1][i]: Y edges for X bin i (dict of 1D arrays)
            - bin_edges[2][i][j]: Z edges for X bin i, Y bin j (nested dict of 1D arrays)
            - info: metadata about binning
    """
    np_x = to_numpy(a)[:,0]
    np_y = to_numpy(a)[:,1]
    np_z = to_numpy(a)[:,2]
    np_sweights = to_numpy(weights) if weights is not None else np.ones_like(np_x)
    
    # 1. Apply range filters to evaluate signal and standard deviations
    mask_x = get_mask(np_x, bins_range[0] if bins_range is not None else None)
    mask_y = get_mask(np_y, bins_range[1] if bins_range is not None else None)
    mask_z = get_mask(np_z, bins_range[2] if bins_range is not None else None)
    global_mask = mask_x & mask_y & mask_z
    
    if not np.any(global_mask):
        raise ValueError("No event left after applying the ranges. Unable to calculate bins.")

    # Filter data by masks
    x_masked = np_x[global_mask]
    y_masked = np_y[global_mask]
    z_masked = np_z[global_mask]
    sweights_masked = np_sweights[global_mask]

    # 2. Calculate the total number of desired 3D bins
    total_yield_in_range = np.sum(sweights_masked)
    if total_yield_in_range <= 0:
        raise ValueError("Total signal (sweights) in the specified range is <= 0.")
    if target_events_per_3d_bin is not None and target_events_per_3d_bin <= 0:
        raise ValueError("target_events_per_3d_bin must be positive.")
    n_total_bins_target = total_yield_in_range / target_events_per_3d_bin

    # 3. Apply the strategy to find n_bins_x, n_bins_y, n_bins_z
    if bins is None:
        if strategy == 'equal':
            if verbose: print(f"Strategy 'equal'. Target N_bins: {n_total_bins_target:.1f}")
            n_base = (n_total_bins_target)**(1/3)
            n_bins_x = max(1, round(n_base))
            n_bins_y = max(1, round(n_base))
            n_bins_z = max(1, round(n_base))
            
        elif strategy == 'proportional':
            if verbose: print(f"Strategy 'proportional'. Target N_bins: {n_total_bins_target:.1f}")
            std_x = weighted_std(x_masked, sweights_masked)
            std_y = weighted_std(y_masked, sweights_masked)
            std_z = weighted_std(z_masked, sweights_masked)
            
            product_std = std_x * std_y * std_z
            if product_std == 0:
                print("Warning: one standard deviation is zero. Falling back to strategy='equal'.")
                return adaptive_bin_edges_3d(
                    a, bins_range, weights, 
                    target_events_per_3d_bin, strategy,
                    verbose=verbose
                )
                
            k = (n_total_bins_target / product_std)**(1/3)
            
            n_bins_x = max(1, round(k * std_x))
            n_bins_y = max(1, round(k * std_y))
            n_bins_z = max(1, round(k * std_z))

        else:
            raise ValueError(f"Strategy '{strategy}' not recognized. Use 'equal' or 'proportional'.")

        if verbose: 
            print(f"Target bins per dimension: X={n_bins_x}, Y={n_bins_y}, Z={n_bins_z}")
            print(f"Total 3D bins (estimated): {n_bins_x * n_bins_y * n_bins_z}")
        bins = (n_bins_x, n_bins_y, n_bins_z)
    else:
        n_bins_x, n_bins_y, n_bins_z = bins
        if verbose: 
            print(f"Using provided bins per dimension: X={n_bins_x}, Y={n_bins_y}, Z={n_bins_z}")
            print(f"Total 3D bins (estimated): {n_bins_x * n_bins_y * n_bins_z}")

    # 4. Recursively create adaptive bin edges
    # Step 4a: Create X bins globally
    if verbose: print("Creating X bin edges (global)...")
    range_x = bins_range[0] if bins_range is not None else None
    range_y = bins_range[1] if bins_range is not None else None
    range_z = bins_range[2] if bins_range is not None else None
    edges_x = adaptive_bin_edges(np_x, weights=np_sweights, bins=n_bins_x, bins_range=range_x)
    n_x_actual = len(edges_x) - 1
    
    # Initialize nested structure for results
    bin_edges = {
        0: edges_x,   # X edges
        1: {},        # Y edges per X bin
        2: {}         # Z edges per (X,Y) bin
    }
    if independent:
        bin_edges[1] = adaptive_bin_edges(np_y, weights=np_sweights, bins=n_bins_y, bins_range=range_y)
        bin_edges[2] = adaptive_bin_edges(np_z, weights=np_sweights, bins=n_bins_z, bins_range=range_z)
    else:
        # Step 4b: For each X bin, create adaptive Y bins
        if verbose: print(f"Creating Y bin edges within each of {n_x_actual} X bins...")
        total_y_bins = 0
        for i in range(n_x_actual):
            # Mask for events in this X bin
            mask_x_bin = (np_x >= edges_x[i]) & (np_x < edges_x[i+1])
            if i == n_x_actual - 1:  # Include upper edge in last bin
                mask_x_bin = (np_x >= edges_x[i]) & (np_x <= edges_x[i+1])
            
            # Apply global range masks
            mask_this_x = mask_x_bin & global_mask
            
            if not np.any(mask_this_x):
                # No events in this X bin, use global Y range
                if verbose: print(f"  X bin {i}: No events, using global Y range")
                bin_edges[1][i] = [range_y[0] if range_y else np_y.min(), 
                                range_y[1] if range_y else np_y.max()]
                bin_edges[2][i] = {}
                continue
                
            # Create adaptive Y bins for this X slice
            edges_y_i = adaptive_bin_edges(
                np_y[mask_this_x], 
                weights=np_sweights[mask_this_x],
                bins=n_bins_y,
                bins_range=range_y
            )
            bin_edges[1][i] = edges_y_i
            n_y_i = len(edges_y_i) - 1
            total_y_bins += n_y_i
            
            # Step 4c: For each (X,Y) bin, create adaptive Z bins
            bin_edges[2][i] = {}
            for j in range(n_y_i):
                # Mask for events in this (X,Y) bin
                mask_y_bin = (np_y >= edges_y_i[j]) & (np_y < edges_y_i[j+1])
                if j == n_y_i - 1:  # Include upper edge in last bin
                    mask_y_bin = (np_y >= edges_y_i[j]) & (np_y <= edges_y_i[j+1])
                
                mask_this_xy = mask_this_x & mask_y_bin
                
                if not np.any(mask_this_xy):
                    # No events in this (X,Y) bin, use global Z range
                    bin_edges[2][i][j] = [range_z[0] if range_z else np_z.min(),
                                        range_z[1] if range_z else np_z.max()]
                    continue
                
                # Create adaptive Z bins for this (X,Y) slice
                edges_z_ij = adaptive_bin_edges(
                    np_z[mask_this_xy],
                    weights=np_sweights[mask_this_xy],
                    bins=n_bins_z,
                    bins_range=range_z
                )
                bin_edges[2][i][j] = edges_z_ij
    
    # 5. Calculate actual total number of 3D bins
    total_3d_bins = 0
    if independent:
        total_3d_bins = (len(bin_edges[0])-1)*(len(bin_edges[1])-1)*(len(bin_edges[2])-1)
    else:
        for i in range(n_x_actual):
            n_y_i = len(bin_edges[1][i]) - 1
            for j in range(n_y_i):
                n_z_ij = len(bin_edges[2][i].get(j, [])) - 1
                total_3d_bins += max(1, n_z_ij)
    
    # 6. Return results in nested structure
    results = {
        'bin_edges': bin_edges,
        'info': {
            'strategy': strategy,
            'target_events_per_3d_bin': target_events_per_3d_bin,
            'total_yield_in_range': total_yield_in_range,
            'n_bins': (n_bins_x, n_bins_y, n_bins_z),
            'total_3d_bins_actual': total_3d_bins,
            'adaptive_structure': 'nested'  # Flag to indicate this is recursive/nested binning
        }
    }
    
    if verbose: 
        print("--- Adaptive 3D Binning Results ---")
        print(f"X axis: {n_x_actual} bins created (requested {n_bins_x})")
        print(f"Y axis: variable bins per X bin (requested {n_bins_y} per slice)")
        print(f"Z axis: variable bins per (X,Y) bin (requested {n_bins_z} per slice)")
        print(f"Total 3D bins: {total_3d_bins} (estimated {n_bins_x * n_bins_y * n_bins_z})")
    
    return results


def get_bin_id(a, binning):
    """Get the bin ID for each event in the array a given the binning.

    Args:
        a (array): array of data to bin.
        binning (array): array of bin edges.

    Returns:
        array: array of bin IDs.
    """    
    npa = to_numpy(a)
    bin_ids = np.digitize(npa, binning) - 1
    # Set candidates outside the bin edges to -1
    bin_ids[(npa < binning[0]) | (npa >= binning[-1])] = -1
    return bin_ids


def get_bin_id_2d(a, binning, independent=False):
    """Get the 2D bin ID for each event in the array a given the binning.

    Args:
        a (array): 2D-array of data to bin.
        binning (dict): dictionary with the bin edges for each dimension.
        independent (bool, optional): independent binning for each axis. Defaults to False.

    Raises:
        ValueError: Input data must be a 2D array with shape (N, 2).

    Returns:
        array: array of 2D bin IDs.
    """    
    npa = to_numpy(a)
    if npa.ndim != 2 or npa.shape[1] != 2:
        raise ValueError("Input data must be a 2D array with shape (N, 2).")
    bin_ids = np.digitize(npa[:,0], binning[0]) - 1
    # Set candidates outside the bin edges to -1
    bin_ids[(npa[:,0] < binning[0][0]) | (npa[:,0] >= binning[0][-1])] = -1
    bin_ids_2d = np.full_like(bin_ids, -1)
    if independent:
        # independent binning: each axis uses the same bin edges (regular grid)
        bin_ids_y = np.digitize(npa[:,1], binning[1]) - 1
        # Set candidates outside the bin edges to -1
        bin_ids_y[(npa[:,1] < binning[1][0]) | (npa[:,1] >= binning[1][-1])] = -1

        # i (x indices) are already in `bin_ids` (with outside set to -1)
        i_arr = bin_ids.copy()

        nb_y = len(binning[1]) - 1

        # Valid entries are those where both indices are >= 0
        valid = (i_arr >= 0) & (bin_ids_y >= 0)
        if np.any(valid):
            bin_ids_2d[valid] = i_arr[valid] * nb_y + bin_ids_y[valid]
        # Entries outside any axis remain -1 in bin_ids_2d
        return bin_ids_2d
    for i in range(len(binning[0])-1):
        mask_x = (npa[:,0] >= binning[0][i]) & (npa[:,0] < binning[0][i+1])
        if i == len(binning[0])-2:  # Include upper edge in last bin
            mask_x = (npa[:,0] >= binning[0][i]) & (npa[:,0] <= binning[0][i+1])
        if np.any(mask_x):
            bin_ids_y = np.digitize(npa[mask_x,1], binning[1][i]) - 1
            # Set candidates outside the bin edges to -1
            bin_ids_y[(npa[mask_x,1] < binning[1][i][0]) | (npa[mask_x,1] >= binning[1][i][-1])] = -1
            bin_ids_2d[mask_x] = i*(len(binning[1][i])-1) + bin_ids_y
    return bin_ids_2d


def get_bin_id_3d(a, binning, independent=False):
    """Get the 3D bin ID for each event in the array a given the binning.

    Args:
        a (array): 3D-array of data to bin.
        binning (dict): dictionary with the bin edges for each dimension.
        independent (bool, optional): independent binning for each axis. Defaults to False.
    
    Raises:
        ValueError: Input data must be a 3D array with shape (N, 3).

    Returns:
        array: array of 3D bin IDs.
    """    
    npa = to_numpy(a)
    if npa.ndim != 2 or npa.shape[1] != 3:
        raise ValueError("Input data must be a 3D array with shape (N, 3).")
    bin_ids = np.digitize(npa[:,0], binning[0]) - 1
    # Set candidates outside the bin edges to -1
    bin_ids[(npa[:,0] < binning[0][0]) | (npa[:,0] >= binning[0][-1])] = -1
    bin_ids_3d = np.full_like(bin_ids, -1)
    if independent:
        # independent binning: each axis uses the same bin edges (regular grid)
        bin_ids_y = np.digitize(npa[:,1], binning[1]) - 1
        bin_ids_z = np.digitize(npa[:,2], binning[2]) - 1
        # Set candidates outside the bin edges to -1 for y and z
        bin_ids_y[(npa[:,1] < binning[1][0]) | (npa[:,1] >= binning[1][-1])] = -1
        bin_ids_z[(npa[:,2] < binning[2][0]) | (npa[:,2] >= binning[2][-1])] = -1

        # i (x indices) are already in `bin_ids` (with outside set to -1)
        i_arr = bin_ids.copy()

        nb_y = len(binning[1]) - 1
        nb_z = len(binning[2]) - 1

        # Valid entries are those where all three indices are >= 0
        valid = (i_arr >= 0) & (bin_ids_y >= 0) & (bin_ids_z >= 0)
        if np.any(valid):
            bin_ids_3d[valid] = (
                i_arr[valid] * (nb_y * nb_z)
                + bin_ids_y[valid] * nb_z
                + bin_ids_z[valid]
            )
        # Entries outside any axis remain -1 in bin_ids_3d
        return bin_ids_3d

    for i in range(len(binning[0])-1):
        mask_x = (npa[:,0] >= binning[0][i]) & (npa[:,0] < binning[0][i+1])
        if i == len(binning[0])-2:  # Include upper edge in last bin
            mask_x = (npa[:,0] >= binning[0][i]) & (npa[:,0] <= binning[0][i+1])
        if np.any(mask_x):
            bin_ids_y = np.digitize(npa[mask_x,1], binning[1][i]) - 1
            # Set candidates outside the bin edges to -1
            bin_ids_y[(npa[mask_x,1] < binning[1][i][0]) | (npa[mask_x,1] >= binning[1][i][-1])] = -1
            for j in range(len(binning[1][i])-1):
                # bin_ids_y is defined only on the subset where mask_x is True.
                # Build a full-length mask_y aligned to the original array.
                mask_y = np.zeros_like(mask_x, dtype=bool)
                mask_y[mask_x] = (bin_ids_y == j)
                if np.any(mask_y):
                    bin_ids_z = np.digitize(npa[mask_y,2], binning[2][i][j]) - 1
                    # Set candidates outside the bin edges to -1
                    bin_ids_z[(npa[mask_y,2] < binning[2][i][j][0]) | (npa[mask_y,2] >= binning[2][i][j][-1])] = -1
                    bin_ids_3d[mask_y] = i*(len(binning[1][i])-1)*(len(binning[2][i][j])-1) +\
                                         j*(len(binning[2][i][j])-1) + \
                                         bin_ids_z
    return bin_ids_3d


def get_bin_from_edges(x, bin_edges):
    """A function that returns an array with the bin index for each candidate in the original array.

    Args:
        x (numpy.ndarray): the array with the data.
        bin_edges (list): list of bin edges.
    
    Returns:
        array: an array with the decay time bin for each candidate.
    """
    bin_indices = np.digitize(x, bin_edges) - 1
    # Set candidates outside the bin edges to -1
    bin_indices[(x < bin_edges[0]) | (x >= bin_edges[-1])] = -1
    return bin_indices