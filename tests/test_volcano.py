import m2aia as m2
from m2aia.utils import volcano
from m2aia.utils import binning
import matplotlib.pyplot as plt
import numpy as np


image = m2.ImzMLReader("/home/jtfc/HS/Data/MTBLS282_LiverSections/47TopRight, 57BottomRight, 67BottomLeft, 77TopLeft/47TopRight, 57BottomRight, 67BottomLeft, 77TopLeft-centroid.imzML")
image.Execute()

# generate a mask by thresholds
mask = image.GetArray(700,2)
tmp = mask > 300
tmp2 = mask > 100

# mask containse labels (0: background, 1: HT (healthy tissue), 2: CT (cancerous tissue))
mask = tmp2.astype(np.int16) + tmp.astype(np.int16)

try:
    bin_ass = np.load('bin_ass.npy')
    bin_xs = np.load('bin_xs.npy') 
    bin_counts = np.load('bin_counts.npy')
except:
    bin_ass, bin_xs, bin_counts = binning.bin_peaks(image, mask)
    bin_counts = [len(x) for x in bin_counts]
    np.save('bin_ass.npy', bin_ass)
    np.save('bin_xs.npy', bin_xs) 
    np.save('bin_counts.npy', bin_counts)

# FILTER PEAKS
# peaks mus occur in more than p*100 of the spectra
p = 0.1
xs = bin_xs[bin_counts > (p*np.sum(mask>0))]
p_fc_sig = volcano.volcano_plot_data(image, mask, xs)

plt.scatter(p_fc_sig[:,1], -np.log10(p_fc_sig[:,0]))
plt.savefig("test.png")



