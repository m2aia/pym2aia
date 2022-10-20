from collections import Counter
import numpy as np


# def grouperRelaxed(xs, ys, sources, tolerance):
#   meanMass = np.mean(xs)

#   ## all peaks in range?
#   if any(np.abs(xs - meanMass) / meanMass > tolerance):
#     return None

#   counts = Counter(sources)
#   ## choose highest peak in dulicates
#   duplicates = [key for key, count in counts.items() if count >1]
#   if any(duplicates):

#     ys_ids = np.argsort(ys[::-1])
#     ys = ys[ys_ids]
#     xs = xs[ys_ids]
#     sources = sources[ys_ids]

#     unique_ids = []
#     for i, s in enumerate(sources):
#         if ~len(duplicates):
#             break
#         if s in duplicates:
#             duplicates.remove(id)
#             unique_ids.append(xs[i])


#     noDup <- !duplicated(samples)

#     noDup[s$ix] <- noDup

#     ## replace mass corresponding to highest intensity
#     mass[noDup] <- mean(mass[noDup])

#     return(mass)
  

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


def group_binning(xs, ys, sorted_sources, grouper, tolerance=0.002):
    """
    xs: sorted list of m/z values of peaks
    ys: sorted list of 
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
        l = grouper(xs[left: gapIdx], ys[left: gapIdx], sorted_sources[left: gapIdx], tolerance)
        if l == None:
            boundary.append((left,gapIdx))
        else:
            bin_assignments[left: gapIdx] = current_id
            x, c = l
            bin_counts.append(c)
            bin_xs.append(x)
            current_id+=1

        # check right interval
        r = grouper(xs[gapIdx: right], ys[gapIdx: right], sorted_sources[gapIdx:right], tolerance)
        if r == None:
            boundary.append((gapIdx, right))
        else:
            bin_assignments[gapIdx:right] = current_id
            x, c = r
            bin_counts.append(c)
            bin_xs.append(x)
            current_id+=1

    tmp = np.zeros_like(bin_assignments)
    for i, k in enumerate(list( dict.fromkeys(bin_assignments.tolist()))):
        tmp[bin_assignments==k] = i #bin id starts by 1

    bin_xs = np.array(bin_xs)
    sort_indices = np.argsort(bin_xs)

    return tmp, bin_xs[sort_indices], [bin_counts[c] for c in sort_indices]
