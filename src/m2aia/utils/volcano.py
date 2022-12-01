import m2aia as m2
import numpy as np
from scipy.stats import ttest_ind
from mne.stats import fdr_correction


def volcano_data(image: m2.ImzMLReader, mask: np.array, x):
    data = image.GetArray(x, x*100*10e-6)
    A = data[mask==1]
    B = data[mask==2]
    return ttest_ind(A,B)[1], np.log2(np.mean(A)/np.mean(B))
    

def volcano_plot_data(image: m2.ImzMLReader, mask: np.array, xs, pThreshold = 0.05, fcThreshold = 1):
    print("Start volcano")
    p_fc_sig = np.zeros((len(xs),3))
    for i, x in enumerate(xs):
        p_fc_sig[i,:2] = volcano_data(image, mask, x)
    p_fc_sig[:,0][p_fc_sig[:,0] == 0] = np.finfo(np.float64).tiny
    p_fc_sig[:,0] = fdr_correction(p_fc_sig[:,0])[1]
    p_fc_sig[:,2] = np.logical_and(p_fc_sig[:,0] < pThreshold, abs(p_fc_sig[:,1]) > fcThreshold)

    return p_fc_sig



