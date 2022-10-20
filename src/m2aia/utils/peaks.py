from ..ImageIO import ImzMLReader
from scipy.signal import find_peaks
from scipy import stats
import numpy as np

def noise_mad(xs, ys):
    return stats.median_abs_deviation(ys, scale=1/1.4826)

def pick_peaks(I: ImzMLReader, 
               SNR: float=8,
               sort_by_xs: bool=True,
               noise_estimator=noise_mad):
    """Pick peaks for a given ImzMLReader object. 
    
    This function is iterating over all spectra in an image and picks peaks.
    sort_by_xs: if True all values of the return arrays are sorted according to ascending peaks_xs values.

    returns:
    peak_xs: list of m/z positions 
        e.g.: peak_xs=[x1 x1 x1 x2 x2 x2 x2 x3 x3 x3 x3 x3 x3]
            3 peaks found in spectrum with id 1
            4 peaks found in spectrum with id 2
            6 peaks found in spectrum with id 3
            
    peak_ys: list of intensities corresponding to peak_xs
        e.g.: peak_ys=[y1 y1 y1 y2 y2 y2 y2 y3 y3 y3 y3 y3 y3]

    peak_idx = list of array indices corresponding to a spectrum array
        e.g.: peak_idx=[idx1 idx1 idx1 idx2 idx2 idx2 idx2 idx3 idx3 idx3 idx3 idx3 idx3]

    peaks_source = indicators for each corresponding peak_xs, from which spectrum id it was picked
        e.g.: peaks_source=[1 1 1 2 2 2 2 3 3 3 3 3 3]

    """
    
    peak_xs = []
    peak_ys = []
    peak_idx = []
    peaks_source = []

    for i, xs, ys in I.SpectrumIterator():

        noise = noise_estimator(ys)
        p = find_peaks(ys, height=noise*SNR)
        
        # masses
        peak_xs.extend(xs[p[0]])
        
        # xs-index for peak
        peak_idx.extend(p[0])

        # intensities
        peak_ys.extend(ys[p[0]])
        
        # spectrum id for each peak
        peaks_source.extend(len(p[0]) * [i])
    
    if sort_by_xs == True:
        sorted_indices = np.argsort(peak_xs)
        peak_xs = np.array(peak_xs)[sorted_indices]
        peak_ys = np.array(peak_ys)[sorted_indices]
        peak_idx = np.array(peak_idx)[sorted_indices]
        peaks_source = np.array(peaks_source)[sorted_indices]
    
    return peak_xs, peak_ys, peak_idx, peaks_source
