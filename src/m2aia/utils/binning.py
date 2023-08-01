from collections import Counter
import numpy as np
import m2aia as m2

## .grouperStrict
##  strict grouping function
##  Don't allow peaks of one sample in the same bin.
##
## params:
##  mass: double, sorted mass
##  intensities: double, corresponding intensities
##  samples: double, corresponding sample id numbers
##  tolerance: double, maximal deviation of a peak position to be
##             considered as same peak
##
## returns:
##  NA if further splitting is needed
##  meanMass (double) if all criteria are matched
def grouper_strict(xs, ys, sources, tolerance):

    counts = Counter(sources)
    if any([key for key, count in counts.items() if count >1]):
        return None

    mean_xs = np.mean(xs)

    if any(np.abs(xs - mean_xs) / mean_xs > tolerance):
        return None

    return mean_xs, counts


def group_binning(xs, ys, ss, grouper, tolerance=0.002):
    """
    xs: sorted list of m/z values of peaks
    ys: sorted list of intensities according to xs sorting
    ss: sorted list of source indices according to xs sorting
    grouper: binning method
    tolerance: TODO

    Returns

    bin_assignments:
        assigns each entry in sorted list to the corresponding bin, so that
        xs[bin_assignments==k] returns all xs values corresponding to the k'th bin
    bin_xs:
        new xs center values of the bins
    counts:
        the count of pixels associated with each peak

    """

    d = xs[1:] - xs[:-1]
    bin_assignments = np.zeros_like(xs,dtype=int)

    n = len(xs)
    boundary = []
    boundary.append((0,n))
    current_id = 0
    bin_counts = []
    bin_xs = []
    
    while len(boundary):
        left, right = boundary.pop()
        gapIdx = np.argmax(d[left:right-1]) + left + 1

        # check left interval
        l = grouper(xs[left: gapIdx], ys[left: gapIdx], ss[left: gapIdx], tolerance)
        if l == None:
            boundary.append((left,gapIdx))
        else:
            bin_assignments[left: gapIdx] = current_id
            x, c = l
            bin_counts.append(c)
            bin_xs.append(x)
            current_id+=1

        # check right interval
        r = grouper(xs[gapIdx: right], ys[gapIdx: right], ss[gapIdx:right], tolerance)
        if r == None:
            boundary.append((gapIdx, right))
        else:
            bin_assignments[gapIdx:right] = current_id
            x, c = r
            bin_counts.append(c)
            bin_xs.append(x)
            current_id+=1

    bin_assign = np.zeros_like(bin_assignments)
    for i, k in enumerate(list( dict.fromkeys(bin_assignments.tolist()))):
        bin_assign[bin_assignments==k] = i #bin id starts by 1

    bin_xs = np.array(bin_xs)
    sort_indices = np.argsort(bin_xs)

    return bin_assign, bin_xs[sort_indices], [bin_counts[c] for c in sort_indices]


def bin_peaks(image: m2.ImzMLReader, mask: np.array = None):
    
    print("Bin peaks for :", image.GetSpectrumType())
    if "ProcessedCentroid" != image.GetSpectrumType():
        raise TypeError("Image has to be of type ProcessedCentroid")

    if mask is not None:
        indices = image.GetIndexArray()[mask>0]
    else:
        indices = image.GetIndexArray()[image.GetMaskArray()>0]

    # BIN PEAKS
    xs_list = [] # list of all mz values
    ys_list = [] # list of all intensities
    ss_list = [] # list of source indices

    for i in indices:
        xs,ys = image.GetSpectrum(i)
        xs_list.extend(xs)
        ys_list.extend(ys)
        ss_list.extend(len(xs) * [i])

    # sort lists
    sort_indices = np.argsort(xs_list)
    xs_list = np.array(xs_list)[sort_indices]
    ys_list = np.array(ys_list)[sort_indices]
    ss_list = np.array(ss_list)[sort_indices]

    print("Start binning on ", len(xs_list), "individual m/z values found in", np.sum(mask>0), "pixels.")

    return group_binning(xs_list, ys_list, ss_list, grouper_strict)
