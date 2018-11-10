from osgeo import gdal
import os
import re
import numpy as np
# import sckit-learn
#EDicion de prueba
import matplotlib.pyplot as plt

def muestra_banda(array):
    """Equalize the image histogram."""
    imflat = np.ravel(array)
    image_histogram, bins = np.histogram(imflat, 256, normed=True)
    cdf = image_histogram.cumsum()
    cdf = 255 * cdf / cdf[-1]
    image_equalized = np.interp(imflat, bins[:-1], cdf)
    image_equalized = image_equalized.reshape(array.shape)
    return image_equalized


images = [f for f in os.listdir()
          if re.match(pattern=r'.*B[3-4]\.TIF', string=f) is not None]
# es el resultado de lo de arriba
# images = ['LT05_L1TP_026047_20010401_20161211_01_T1_B3.TIF',
#           'LT05_L1TP_026047_20010401_20161211_01_T1_B4.TIF']
image = gdal.Open(images[0], gdal.GA_ReadOnly)
concatenate = np.empty((image.RasterYSize*image.RasterXSize, 0))
for imtitle in images:
    image = gdal.Open(imtitle, gdal.GA_ReadOnly)
    # image.GetDescription()
    # image.GetDriver()
    # driver.GetDescription()
    # driver.GetMetadata()
    # driver.GetMetadata_Dict()
    # driver.GetMetadataItem()
    # image.GetGeoTransform()
    # driver = image.GetDriver()
    # image.GetGCPProjection()
    # image.GetRasterBand(1)
    print(image.RasterCount, image.RasterXSize, image.RasterYSize)
    band = image.GetRasterBand(1)
    array = band.ReadAsArray()
    concatenate = np.hstack(tup=(concatenate, array.reshape((-1,1))))
    fig = plt.figure()
    plt.imshow(muestra_banda(array), cmap='Reds')
    plt.colorbar()
plt.show()
print(concatenate.shape)
print(concatenate[:,0].min())
print(concatenate[:,0].max())
print(concatenate[:,0].std())
print(concatenate[:,0].mean())
plt.hist(concatenate[:,0]); plt.show()
plt.hist(concatenate[:,1]); plt.show()
# Para guardar en .npz el arreglo bajo la llave concatenate
np.savez('resultado', concatenate=concatenate)
# Para leerla seria:
cosa = np.load('resultado.npz')
print(cosa['concatenate'])
