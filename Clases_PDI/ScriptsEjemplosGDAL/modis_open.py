from osgeo import gdal
imag = gdal.Open('HDF4_EOS:EOS_SWATH:"MOD021KM.A2012111.1645.005.2012112014505.hdf":MODIS_SWATH_Type_L1B:EV_1KM_Emissive_Uncert_Indexes')
band = imag.GetRasterBand(1)
imag.GetGeoTransform()
imag.GetProjection()
from osgeo import ogr
from osgeo import osr
wgs84 = osr.SpatialReference()
wgs84.ImpoertFromEPSG(4326)
wgs84.ImportFromEPSG(4326)
wgs84
dir(wgs84)
modis_sinu = osr.SpatialReference()
modis_sinu.ImportFromProj4("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
tx = osr.CoordinateTransformation(wgs84, modis_sinu)
tx
lon, lat = (-3.904, 50.58)
modis_x, modis_y, modis_z = tx.TransformPoint(lon, lat)
modis_x
modis_y
modis_z
%hist -f modis_open.py
