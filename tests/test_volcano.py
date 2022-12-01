import m2aia as m2
from m2aia.utils import volcano
from m2aia.utils import binning
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sea
import seaborn_image as seai
import SimpleITK as sitk

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
print("Binned peaks found", len(bin_xs))

# FILTER PEAKS
# peaks mus occur in more than p*100 of the spectra
p = 0.1
xs = bin_xs[bin_counts > (p*np.sum(mask>0))]
print("Filtered peaks found", len(bin_xs))

p_fc_sig = volcano.volcano_plot_data(image, mask, xs)

print("significant", np.sum(p_fc_sig[:,2]))
sea.scatterplot(x=p_fc_sig[:,1], y=-np.log10(p_fc_sig[:,0]),hue=p_fc_sig[:,2],legend=["-", "Significant"])
plt.savefig("test.png", dpi=300)
images = []
for x in xs[np.logical_and(p_fc_sig[:,2] == 1,p_fc_sig[:,1] > 3) ]:
    images.append(image.GetArray(x,100*x*10e-6)[0])

seai.ImageGrid(images[:10])
plt.savefig("testimages.png", dpi=300)



