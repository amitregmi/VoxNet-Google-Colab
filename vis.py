import os
import numpy as np
import binvox_rw

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

strPath = os.path.join("C:\Users\Amit_Regmi\Desktop\Amit2015\Artificial Intelligence\Python Scripts\modelnet-cnn-master\data_prep\\binvox\\toilet_obj_30.binvox")
data = open(strPath, 'rb') 
voxel = binvox_rw.read_as_3d_array(data, fix_coords = True)

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.voxels(voxel, edgecolor='k')
Axes3D(fig).scatter(*np.nonzero(voxel.data))
plt.show()