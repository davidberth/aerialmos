from PIL import Image
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import sys
from numba import jit, void, int64, int32, float64
import pandas as pd
import geopandas as gpd
#import contextily as ctx


# be sure to do bounds checks on the variables before passing them into this function
@jit(void(int32[:,:,:],int32[:,:],int32[:,:],float64[:], float64[:], float64[:],
          float64[:],float64[:],int64,int64,int64,int64), nopython=True)
def optimize(tiles, im, out, inx, iny, outx, outy, outl, tsize, ntiles, ox, oy):

        e = 0
        ll = 0
        actdiffmean = 0
        for y in range(0, oy):
            print(y, oy)
            for x in range(0, ox):


                # here we determine ll
                llmin = 9999999999
                xx = x * tsize
                yy = y * tsize

                comp = im[yy:yy + tsize, xx:xx + tsize]

                for tile in range(ntiles):
                    diff = tiles[tile, :, :] - comp
                    diffmean = int(np.mean(diff))
                    diffabs = np.sum(np.absolute(tiles[tile, :, :] - comp - diffmean))
                    if diffabs < llmin:
                        llmin = diffabs
                        ll = tile
                        actdiffmean = int(diffmean)
                        outx[e] = inx[tile]
                        outy[e] = iny[tile]
                        outl[e] = diffabs

                out[yy:yy + tsize, xx:xx + tsize] = tiles[ll, :, :] - actdiffmean
                e += 1

tsize = 22
np.random.seed(44)





im = Image.open(sys.argv[1])


im = np.array(im)

ntiles = int(sys.argv[2])

im = im[:, :, 0] // 3 + im[:, :, 1] //3  + im[:, : ,2] //3
im = im.astype(np.int32)
#scenes = ['c:/satmos/imagery/a.tif', 'c:/satmos/imagery/b.tif']
scenes = ['c:/satmos/imagery/aa.tif']
scs = []

i = scenes[0]
scene = gdal.Open(i)
ulx, xres, xskew, uly, yskew, yres  = scene.GetGeoTransform()

band1 = np.array(scene.GetRasterBand(1).ReadAsArray())
band2 = np.array(scene.GetRasterBand(2).ReadAsArray())
band3 = np.array(scene.GetRasterBand(3).ReadAsArray())
sc = band1 //3  + band2 //3 + band3 //3
sc-= (sc.min())
sc = sc.astype(np.float32) * ( im.max() - im.min()) / (sc.max() - sc.min())
sc+= (im.min())

sc = sc.astype( np.int32)
print (sc.min(), sc.max())
print (im.min(), im.max())
print(sc.shape)
print(im.shape)

plt.imshow(np.dstack( (im, im, im)))
plt.show()


print ('generating tiles')

tiles = np.zeros((ntiles, tsize, tsize), dtype=np.int32)

inx = np.zeros(ntiles)
iny = np.zeros(ntiles)
for tile in range(ntiles):
        xx = np.random.randint(0, sc.shape[1] - tsize - 1)
        yy = np.random.randint(0, sc.shape[0] - tsize - 1)

        tiles[tile, :, :] = sc[yy:yy + tsize, xx:xx+tsize]
        inx[tile] = xx
        iny[tile] = yy

print ('done')


oy = int(im.shape[0] / tsize - 1)
ox = int(im.shape[1] / tsize - 1)
otiles = oy * ox

outx = np.zeros(otiles)
outy = np.zeros(otiles)
outl = np.zeros(otiles)

out = np.zeros((oy * tsize, ox * tsize), np.int32)

print ('generating output')

optimize(tiles, im, out, inx, iny, outx, outy, outl, tsize, ntiles, ox, oy)
out[out<0] = 0
out[out>255] = 255
out = out.astype(np.uint8)


px = ulx + (inx * xres)
py = uly + (iny * yres)

print (px)
print (py)


im = Image.fromarray( np.dstack(( out, out, out )))
im.save("c:/satmos/out/out.jpeg")

#plt.imshow( np.dstack( [out, out, out]))
#plt.show()

#df = pd.DataFrame( zip(px, py, outl), columns = ['lon', 'lat', 'outl'])
#df.to_csv('coordinates/1.csv')


#df['outl'] /= np.max(df['outl'])


##gdf = gpd.GeoDataFrame(
#    df, geometry=gpd.points_from_xy(df.lon, df.lat))
#gdf.set_crs(epsg=4326, inplace=True)
#gdf = gdf.to_crs(epsg=3857)




#ax = gdf.plot(markersize=7,  cmap='Reds', column='outl', scheme='percentiles', figsize=(12,12), legend=True,
#              legend_kwds={'title':'Normalized Error'})
#plt.title('Clip Samples')

#ctx.add_basemap(ax)

#plt.savefig('c:/satmos/coordinates/out.png')

