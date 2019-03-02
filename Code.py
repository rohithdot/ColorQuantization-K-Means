import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import sys

from sklearn import cluster



def quantize(raster, n_colors):
    width, height, depth = raster.shape
    reshaped_raster = np.reshape(raster, (width * height, depth))

    model = cluster.KMeans(n_clusters=n_colors)
    labels = model.fit_predict(reshaped_raster)
    palette = model.cluster_centers_

    quantized_raster = np.reshape(palette[labels], (width, height, palette.shape[1]))

    return quantized_raster


raster = scipy.misc.imread(sys.argv[1])

k=int(sys.argv[2])
temp=quantize(raster,k)


plt.imshow(temp/ 255.0)
plt.draw()
plt.show()