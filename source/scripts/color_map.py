import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


colors = ["#8bcc78", "#29aae1","#ff7f50","#7030a0"]
colors2 = ["#1d6309", "#0d4b92", "#a92e01","#301545"]
c = mcolors.ColorConverter().to_rgb
seq=[c('white'),c(colors[1]),0.33,c(colors[1]),c(colors2[1]),0.66,c(colors2[1])]
rvb = make_colormap(seq)
N = 1000
array_dg = np.random.uniform(0, 10, size=(N, 2))
colors = np.random.uniform(-2, 2, size=(N,))
plt.scatter(array_dg[:, 0], array_dg[:, 1], c=colors, cmap=rvb)
plt.colorbar()
plt.show()